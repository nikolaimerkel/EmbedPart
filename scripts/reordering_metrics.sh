echo 0/4 
python scripts/reordering_metrics.py --absolute_path /mnt/reordered/ogbn-arxiv.directed.cuttana.cuttana256.P32.reordered --results_dir /mnt/data/gnn-partitioning/results/reordering-metrics/2026-01-05_14-48-48
echo 1/4 
python scripts/reordering_metrics.py --absolute_path /mnt/reordered/ogbn-papers100M.directed.cuttana.cuttana256.P32.reordered --results_dir /mnt/data/gnn-partitioning/results/reordering-metrics/2026-01-05_14-48-48
echo 2/4 
python scripts/reordering_metrics.py --absolute_path /mnt/reordered/ogbn-products.directed.cuttana.cuttana256.P32.reordered --results_dir /mnt/data/gnn-partitioning/results/reordering-metrics/2026-01-05_14-48-48
echo 3/4 
python scripts/reordering_metrics.py --absolute_path /mnt/reordered/reddit.directed.cuttana.cuttana256.P32.reordered --results_dir /mnt/data/gnn-partitioning/results/reordering-metrics/2026-01-05_14-48-48
