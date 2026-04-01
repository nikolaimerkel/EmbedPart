
 echo 0/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-arxiv.metis.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/1

 echo 1/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-arxiv.random.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/2

 echo 2/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-arxiv.ldg.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/3

 echo 3/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-arxiv.spinner.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/4

 echo 4/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/5

 echo 5/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/6

 echo 6/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-products.metis.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/7

 echo 7/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-products.random.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/8

 echo 8/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-products.ldg.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/9

 echo 9/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-products.spinner.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/10

 echo 10/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-products.Dogbn-products.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/11

 echo 11/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-products.Dogbn-products.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/12

 echo 12/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/reddit.metis.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/13

 echo 13/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/reddit.random.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/14

 echo 14/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/reddit.ldg.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/15

 echo 15/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/reddit.spinner.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/16

 echo 16/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/reddit.Dreddit.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/17

 echo 17/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/reddit.Dreddit.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.rerordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/18

 echo 18/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-arxiv.directed.cuttana.cuttana256.P32.reordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/19

 echo 19/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-products.directed.cuttana.cuttana256.P32.reordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/20

 echo 20/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/reddit.directed.cuttana.cuttana256.P32.reordered -infrastructure cuda -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/21

 echo 21/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-arxiv.metis.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/22

 echo 22/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-arxiv.random.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/23

 echo 23/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-arxiv.ldg.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/24

 echo 24/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-arxiv.spinner.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/25

 echo 25/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/26

 echo 26/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/27

 echo 27/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-products.metis.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/28

 echo 28/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-products.random.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/29

 echo 29/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-products.ldg.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/30

 echo 30/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-products.spinner.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/31

 echo 31/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-products.Dogbn-products.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/32

 echo 32/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-products.Dogbn-products.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/33

 echo 33/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/reddit.metis.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/34

 echo 34/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/reddit.random.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/35

 echo 35/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/reddit.ldg.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/36

 echo 36/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/reddit.spinner.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/37

 echo 37/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/reddit.Dreddit.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/38

 echo 38/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/reddit.Dreddit.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/39

 echo 39/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-papers100M.metis.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/40

 echo 40/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-papers100M.random.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/41

 echo 41/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-papers100M.ldg.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/42

 echo 42/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-papers100M.spinner.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/43

 echo 43/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/44

 echo 44/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-papers100M.Dogbn-papers100M.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1.P32.rerordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/45

 echo 45/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-papers100M.directed.cuttana.cuttana256.P32.reordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/46

 echo 46/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-arxiv.directed.cuttana.cuttana256.P32.reordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/47

 echo 47/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/ogbn-products.directed.cuttana.cuttana256.P32.reordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/48

 echo 48/49 
python3 train.py -absolute_path_to_graph /mnt/reordered/reddit.directed.cuttana.cuttana256.P32.reordered -infrastructure cpu -num_epochs 10 -num_layers 2 -hidden_dim 64 -results_dir /mnt/data/gnn-partitioning/results/single-node-training/2026-01-05_20-00-41/49
