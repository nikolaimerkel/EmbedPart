# EmbedPart: Embedding-Driven Graph Partitioning for Scalable Graph Neural Network Training

Graph Neural Networks (GNNs) are widely used for learning on graph-structured data, but scaling GNN training to massive graphs remains challenging. To enable scalable distributed training, graphs are divided into smaller partitions that are distributed across multiple machines such that inter-machine communication is minimized and computational load is balanced. In practice, existing partitioning approaches face a fundamental trade-off between partitioning overhead and partitioning quality.

We propose EmbedPart, an embedding-driven partitioning approach that achieves both speed and quality. Instead of operating directly on irregular graph structures, EmbedPart leverages node embeddings produced during the actual GNN training workload and clusters these dense embeddings to derive a partitioning. 

EmbedPart achieves more than 100× speedup over Metis while
maintaining competitive partitioning quality and accelerating distributed GNN training. Moreover, EmbedPart naturally supports graph updates and fast repartitioning, and can be applied to graph reordering to improve data locality and accelerate single-machine GNN training. By shifting partitioning from irregular graph structures to dense embeddings, EmbedPart enables scalable and highquality graph data optimization.

Work currently in submission.
Contact: <nikolai.merkel@tum.de>

# Installation:

Setup (Distributed) DGL. 
We need a NFS for the workers.

```bash
# NFS master / DGL naster
sudo dnf install nfs-utils
sudo vim /etc/exports
# Put this in (for each worker write replace xxx with the ip)
/data/sdb/nikolai xxx.xxx.xxx.xxx(rw,sync,no_subtree_check,no_root_squash)
/data/sdb/nikolai xxx.xxx.xxx.xxx(rw,sync,no_subtree_check,no_root_squash)
/data/sdb/nikolai xxx.xxx.xxx.xxx(rw,sync,no_subtree_check,no_root_squash)
/data/sdb/nikolai xxx.xxx.xxx.xxx(rw,sync,no_subtree_check,no_root_squash)
sudo exportfs -rav
sudo systemctl restart nfs-server
sudo firewall-cmd --permanent --add-service=nfs
sudo firewall-cmd --reload
sudo systemctl enable nfs-server
sudo systemctl start nfs-server
sudo chmod 777 -R /data/sdb/nikolai

# NFS client / DL worker
sudo dnf install nfs-utils
sudo mkdir -p /data/sdb/nikolai
sudo mount xxx.xxx.xxx.xxx:/data/sdb/nikolai /data/sdb/nikolai
sudo chmod 777 -R /data/sdb/nikolai
```
Build the master container
```bash
# Build DGL master
sudo docker build -t dgl-master -f docker/Dockerfile.dgl-master-nvidia .
sudo docker run -d --network host --gpus all --name dgl-master -v /data/sdb/nikolai/gnn-partitioner:/mnt/data -v /data/sda/nikolai/reordered:/mnt/reordered --shm-size=256G dgl-master
```

connect to master to get the pub ssh key of the master
```bash
sudo docker exec -it dgl-master /bin/sh -c "cat /root/.ssh/id_ed25519.pub"
```

Add key to worker nodes. It is needed **before** the docker worker is build and needs to be in the directory [dgl-bench/docker](dgl-bench/docker) in which the docker build command is executed, from there it will be copied to the container. 

```bash
echo "your key from above" > docker/id_ed25519.pub
```

Build worker container
```bash
sudo docker build -t dgl-worker -f docker/Dockerfile.dgl-worker-nvidia .
```

Run the worker
```bash
sudo docker run -d --network host --gpus all --name dgl-worker -v /data/sdb/nikolai/gnn-partitioner:/mnt/data --shm-size=256G dgl-worker
```

Connecte to master / worker
```sh
sudo docker exec -it dgl-worker /bin/sh
sudo docker exec -it dgl-master /bin/sh
```

Adjust your ips in:
- dgl-bench/ip_config-2.txt
- dgl-bench/ip_config-4.txt

Then connect to master:
```bash 
sudo docker exec -it dgl-master /bin/sh
```

Create scripts
```bash
cd dgl-bench
# Create training script
python create_benchmark.py
# Run it
bash create_benchmark.sh
```

For reordering experiments
```bash
# Create training script
python create_benchmark_single_node.py
# Run it
bash create_benchmark_single_node.sh
```

Download datasets (do everything below in the master container): 

Set some configurations. 

There you can define where the raw dgl graphs, edgelists, and partitioned/reordered graphs will be stored. 

```bash
configs/config.py
```

Download graphs:
```
download_graphs.sh
```

Create also some smaller graphs with:
```bash
# Define sampling ratio: In the paper we use 0.1, 0.3, 0.5, 0.7, 0.9. 
# Create a script with 
python scripts/sparsify.py

# Than run the script with
bash scripts/sparsify.sh
```

Partition with METIS, etc.
```bash
# Create shell script
scripts/partition.py
# Run the created shell script
bash scripts/partition.sh
# Partitioning metrics will be in results/partitioning-metrics/<timstamp>/
# The partitioned grapsh will be store in the directoy defined in `configs/config.py` 
```

Use our approach with: 
```bash
# Create shell scripts
python scripts/embeddings.py
# Run the created shell script
bash scripts/embeddings.sh
# Partitioning metrics will be store in results/partitioning-metrics-fast/<timstamp>/
```

For reordering we need the mapping of vertex ids to partitions ids (vid2pid file).

The file has number of vertices many lines and in each line we have the partitioning number.

For example:
```
8
7
6
...
...
```
Means vertex 0,1, and 2 is in partition 8,6, and 6, respectively.

Rerorder DGL
```bash
# Define the vid2 pid files
# With the following command reorder the graphs
python reordering/reorder_cmds.py
```

Create training scripts for reoreringer: 
```bash
# Create shell script for training or use benchmark script from above
python reordering/train_cmds.py
# Run it 
bash reordering/train.sh
```
Training incl. checkpointing and storing embeddings
```bash
python scripts/train.py
bash scripts/train.sh
```



# Plots
Finally plots the results: 

Install python libs and jupyter notebook for plotting

```bash
python -m venv myenv
source myenv/bin/activate
pip install numpy pandas seaborn matplotlib networkx pyarrow tqdm
source myenv/bin/activate
```

Start jupyter noteboook
```bash
source myenv/bin/activate
```

Run notebooks/Evaluation.ipynb to create plots / tables for the paper 

Plots will be store in [plots](notebooks/plots)
