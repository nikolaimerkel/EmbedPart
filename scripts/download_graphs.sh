echo ogbn-arxiv
python -m  scripts.download_graphs -graph_name ogbn-arxiv
echo ogbn-products
python -m  scripts.download_graphs -graph_name ogbn-products
echo reddit
python -m  scripts.download_graphs -graph_name reddit
echo ogbn-papers100M
python -m  scripts.download_graphs -graph_name ogbn-papers100M
echo ogbl-citation2
python -m  scripts.download_graphs -graph_name ogbl-citation2

