import argparse
from datetime import datetime
import time
    
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
import tqdm


from dgl.data import AsNodePredDataset
from dgl.dataloading import (
    DataLoader,
    MultiLayerFullNeighborSampler,
    NeighborSampler,
)
from ogb.nodeproppred import DglNodePropPredDataset
from partitioner.Partitioner import *
from models.model_factory import get_model
import json
import numpy as np

from training.train_link import to_bidirected_with_reverse_mapping
from training.eval import evaluate
from utils.graph_utils import load_dgl_graph
from models.model_factory import get_model
from models.utils import save_model, load_model

from configs.config import PARTITIONED_GRAPHS

def store_partition_mapping(vid2pid, dir, file):
    
    print(f"store mapping at {dir}")
    
    os.makedirs(dir, exist_ok=True)  # Ensure directory exists
    np.savetxt(f"{dir}/{file}", vid2pid, fmt="%s")


def store_metrics(args, metrics):
    m = {key: int(value) if isinstance(value, np.integer) else value for key, value in metrics.items()}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    with open(f"{args.results_dir}/{timestamp}.json", "w") as f:
        json.dump(m, f)  
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
        required=False
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="graphsage",
        help="The model to train.",
        required=False
    )
     
    
    parser.add_argument(
        "--dataset_to_partition",
        type=str,
        help="the graph that will be partitioned",
        required=True
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        help="the graph that was used to train the GNN model.",
        required=False
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024*8,
        help="The batch size to use.",
        required=False
    )
    
    parser.add_argument(
        "--hidden_dims",
        type=int,
        help="the hidden dimension which is used in the GNN model",
        required=False
    )
    
    parser.add_argument(
        "--out_size",
        type=int,
        help="the output size which is used in the GNN model",
        required=False
    )
    
    parser.add_argument(
        "--num_layers",
        type=int,
        help="the number of layers which is used in the GNN model",
        required=False
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        help="the number of epochs for which the GNN model was trained. 0 means it is a random model",
        required=False
    )
    
    parser.add_argument(
        "--partitioner",
        type=str,
        choices=["gnn-projection", "k-means", "hyperplanes", "random", "metis", "feature-partitioning"],
        help="The partitioner to use.",
        required=True
    )
      
    parser.add_argument(
        '--fanout', 
        nargs='+', 
        type=int, 
        help="was used for training and also will be used for inference",
        required=False
    )
    
    parser.add_argument(
        '--num_parts', 
        nargs='+', 
        type=int, 
        help="The number of partitions to partition the graph",
        required=True
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        help="where to store the metrics. eg results",
        required=True
    )
    
    parser.add_argument(
        "--max_training_balance",
        type=float,
        help="balance of training vertices",
        default=1.05,
        required=False
    )
    
    parser.add_argument(
        "--max_vertex_balance",
        type=float,
        help="max_balance of vertices",
        default=1.1,
        required=False
    )
    
    parser.add_argument(
        "--not_store",
        type=bool,
        default=False,
        help="whether to store the partition mapping",
        required=False
    )
     
    
    args = parser.parse_args()
    
    print(args)
    
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")
    
    print(f"Start loading dataset {args.dataset_to_partition} which should be partitioned.")
    g, num_classes = load_dgl_graph(args.dataset_to_partition)
    print(g)
    print(f"Stop loading dataset {args.dataset_to_partition} which should be partitioned.")
            # for linkprediction where we do not have train_mask
    if 'train_mask' not in g.ndata:
        print("we do not have train_mask in the graph, we create a dummy one")
        g.ndata['train_mask'] = torch.ones(g.num_nodes(), dtype=torch.int)

    
    if args.partitioner == "random":
        metrics_list = []
        print("Perform random partitioning.")
        partitioner = Partitioner(compute_metrics=False)
        for p in args.num_parts:
            start_time = time.time()
            vid2pid, _ = partitioner.random_partitioning(graph=g, graph_name=args.dataset_to_partition,num_parts=p, cache_disc=False)
            stop_time = time.time()
            metrics = partitioner.partition_metricsmetrics(graph=g, vid2pid=vid2pid, num_parts=p, strategy="random")
            metrics["partitioning_time"] = stop_time - start_time
            metrics["graph"] = args.dataset_to_partition
            metrics_list.append(metrics)
            print(metrics)
            if args.not_store == False:    
                store_partition_mapping(vid2pid=vid2pid,  dir=PARTITIONED_GRAPHS, file=f"{args.dataset_to_partition}.random.P{p}.vid2pid")
                store_metrics(args=args, metrics=metrics)
        
        exit()
        
    if args.partitioner == "metis":
        metrics_list = []
        partitioner = Partitioner(compute_metrics=False)
        print("We perform metis partitioning.")

        for p in args.num_parts:
            start_time = time.time()
            vid2pid, _ = partitioner.metis_partitioning(graph=g, graph_name=args.dataset_to_partition,num_parts=p, cache_disc=False)
            stop_time = time.time()
            metrics = partitioner.partition_metricsmetrics(graph=g, vid2pid=vid2pid, num_parts=p, strategy="metis")
            metrics["partitioning_time"] = stop_time - start_time
            metrics["graph"] = args.dataset_to_partition
            metrics_list.append(metrics)
            print(metrics)
           # if args.not_store == False: 
             #   store_partition_mapping(vid2pid=vid2pid, dir=PARTITIONED_GRAPHS, file=f"{args.dataset_to_partition}.metis.P{p}.vid2pid")
              #  store_metrics(args=args, metrics=metrics)
        exit()
    
    if args.partitioner == "feature-partitioning":
        metrics_list = []
        partitioner = Partitioner(compute_metrics=False)
        print("We perform feature partitioning.")
        for p in args.num_parts:
            start_time = time.time()
            vid2pid, _ = partitioner.kmeans(graph=g, embeddings=g.ndata['feat'].numpy(), num_parts=p, train_balance=args.max_training_balance, vertex_balance=args.max_vertex_balance)  
            stop_time = time.time()
            metrics = partitioner.partition_metricsmetrics(graph=g, vid2pid=vid2pid, num_parts=p, strategy="feature-partitioning")
            metrics["partitioning_time"] = stop_time - start_time
            metrics["graph"] = args.dataset_to_partition
            metrics_list.append(metrics)
            print(metrics)
            if args.not_store == False: 
                store_partition_mapping(vid2pid=vid2pid, dir=PARTITIONED_GRAPHS, file=f"{args.dataset_to_partition}.feature-partitioning.P{p}.vid2pid")
                store_metrics(args=args, metrics=metrics)
        exit()
        
    if args.model_name == "linkgraphsage":
        print("We perform link prediction with LinkGraphSage.")
        g = dgl.remove_self_loop(g)
        g, _ = to_bidirected_with_reverse_mapping(g)
 
    
        g = g.to("cuda" if args.mode == "puregpu" else "cpu")
        device = torch.device("cpu" if args.mode == "cpu" else "cuda")
        model = get_model(
            args.model_name, 
            g.ndata["feat"].shape[1], 
            args.hidden_dims, 
            args.out_size, 
            args.num_layers
        )
        
        
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        model, optimizer = load_model(model, opt, args.epochs, args.model_name, args.dataset, args.hidden_dims, args.num_layers, args.fanout,args.out_size)
        name = f"D{args.dataset}.E{args.epochs}.{args.model_name}.H{args.hidden_dims}.L{args.num_layers}.O{args.out_size}.F{'-'.join(map(str, args.fanout))}.TB{args.max_training_balance}.VB{args.max_vertex_balance}"
        model.eval()
        embeddings = None
        time_to_compute_embeddings = None
        metrics_list = []
        with torch.no_grad():  
            time_to_compute_embeddings_start = time.time()
            embeddings = model.inference(g, device, args.batch_size).cpu().numpy()
            time_to_compute_embeddings = time.time() - time_to_compute_embeddings_start
            partitioner =  Partitioner(compute_metrics=False)
            print("loading graph again, as the other graph is bidirected for link prediction")
            g, num_classes = load_dgl_graph(args.dataset_to_partition)
            # for linkprediction where we do not have train_mask
            if 'train_mask' not in g.ndata:
                print("we do not have train_mask in the graph, we create a dummy one")
                g.ndata['train_mask'] = torch.ones(g.num_nodes(), dtype=torch.int)

            for p in args.num_parts:
                vid2pid= None
                if args.partitioner == "k-means":
                    print(f"We partition with K-Means into {p} partitions")    
                    start_time = time.time()   
                    vid2pid, _ = partitioner.kmeans(graph=g, embeddings=embeddings, num_parts=p, train_balance=args.max_training_balance, vertex_balance=args.max_vertex_balance)  
                    stop_time = time.time()
                    metrics = partitioner.partition_metricsmetrics(graph=g, vid2pid=vid2pid, num_parts=p, strategy="kmeans")
                    metrics["partitioning_time"] = stop_time - start_time
                    metrics["compute_embeddings_time"] = time_to_compute_embeddings
                    metrics.update(vars(args) )
                    metrics["model_name"] = args.model_name
                    metrics["graph"] = args.dataset_to_partition
                    metrics["num_parts"] = p
                    metrics_list.append(metrics)
                    print(metrics)
                    fn = f"{args.dataset_to_partition}.{name}.P{p}.vid2pid"
                    print("\nWe would store at the path", PARTITIONED_GRAPHS, fn, "\n")
                    if args.not_store == False: 
                        print("store partition mapping")
                        store_partition_mapping(vid2pid=vid2pid, dir=PARTITIONED_GRAPHS, file=fn)
                        store_metrics(args=args, metrics=metrics)
        print(metrics_list)
        exit()

    if args.model_name in ["graphsage", "gat"]:
        print("We perform node classification with GraphSage or GAT.")
        g = g.to("cuda" if args.mode == "puregpu" else "cpu")
        device = torch.device("cpu" if args.mode == "cpu" else "cuda")
        model = get_model(
            args.model_name, 
            g.ndata["feat"].shape[1], 
            args.hidden_dims, 
            num_classes, 
            args.num_layers
        )
        model.to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        model, optimizer = load_model(model, opt, args.epochs, args.model_name, args.dataset, args.hidden_dims, args.num_layers, args.fanout)
        name = f"D{args.dataset}.E{args.epochs}.{args.model_name}.H{args.hidden_dims}.L{args.num_layers}.F{'-'.join(map(str, args.fanout))}.TB{args.max_training_balance}.VB{args.max_vertex_balance}"
        model.eval()
        embeddings = None
        time_to_compute_embeddings = None
        metrics_list = []
        with torch.no_grad():  
            time_to_compute_embeddings_start = time.time()
            embeddings = model.inference_sampling(model, g, device, args.batch_size,args.fanout).cpu().numpy()
            time_to_compute_embeddings = time.time() - time_to_compute_embeddings_start
            partitioner =  Partitioner(compute_metrics=False)
            for p in args.num_parts:
                vid2pid= None
                if args.partitioner == "k-means":
                    print(f"We partition with K-Means into {p} partitions")    
                    start_time = time.time()   
                    vid2pid, _ = partitioner.kmeans(graph=g, embeddings=embeddings, num_parts=p, train_balance=args.max_training_balance, vertex_balance=args.max_vertex_balance)  
                    stop_time = time.time()
                    metrics = partitioner.partition_metricsmetrics(graph=g, vid2pid=vid2pid, num_parts=p, strategy="kmeans")
                    metrics["partitioning_time"] = stop_time - start_time
                    metrics["compute_embeddings_time"] = time_to_compute_embeddings
                    metrics.update(vars(args) )
                    metrics["model_name"] = args.model_name
                    metrics["graph"] = args.dataset_to_partition
                    metrics["num_parts"] = p
                    metrics_list.append(metrics)
                    print(metrics)
                    fn = f"{args.dataset_to_partition}.{name}.P{p}.vid2pid"
                    print("\nWe would store at the path", PARTITIONED_GRAPHS, fn, "\n")
                    if args.not_store == False: 
                        print("store partition mapping")
                        store_partition_mapping(vid2pid=vid2pid, dir=PARTITIONED_GRAPHS, file=fn)
                        store_metrics(args=args, metrics=metrics)
        print(metrics_list)
        exit()
                    
    
