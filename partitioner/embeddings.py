def kmeans(partitioner, graph, embeddings, num_parts, niter, nredo, max_points_per_centroid, args ):
    t1 = time.time()
    vid2pid, additional_metrics = partitioner.kmeans(
        graph=graph,
        embeddings=embeddings,
        num_parts=num_parts,
        niter=niter,
        verbose=True,
        spherical=True,
        nredo=nredo,
        train_balance=args.max_training_balance,
        vertex_balance=args.max_vertex_balance,
        max_points_per_centroid=max_points_per_centroid,
    )
    t2 = time.time()
    partitioning_time = t2 - t1

    metrics = partitioner.partition_metricsmetrics(graph=graph, vid2pid=vid2pid, num_parts=num_parts, strategy="kmeans")
    metrics["partitioning_time"] = partitioning_time
    metrics.update(vars(args))
    metrics.update(additional_metrics)
    metrics["model_name"] = args.model_name
    metrics["graph"] = args.dataset_to_partition
    metrics["num_parts"] = num_parts
    metrics["niter"] = niter
    metrics["max_points_per_centroid"] = max_points_per_centroid
    metrics["nredo"] = nredo

    return vid2pid, metrics


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
import faiss


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
    os.makedirs(args.results_dir, exist_ok=True)
    with open(f"{args.results_dir}/{timestamp}.json", "w") as f:
        json.dump(m, f)  
            
def compute_or_load_linkgraphsage_embeddings(args):
    name = f"D{args.dataset}.E{args.epochs}.{args.model_name}.H{args.hidden_dims}.L{args.num_layers}.O{args.out_size}.F{'-'.join(map(str, args.fanout))}"
    embeddings = None
    print(f"/mnt/reordered/embeddings/{name}.npy")
    print(f"{name}.npy")
    if f"{name}.npy" in os.listdir("/mnt/reordered/embeddings"):
        embeddings = np.load(f"/mnt/reordered/embeddings/{name}.npy")
    else: 
        g, num_classes = load_dgl_graph(args.dataset_to_partition)
        print(g)
        print(f"Stop loading dataset {args.dataset_to_partition} which should be partitioned.")
                    # for linkprediction where we do not have train_mask
        if 'train_mask' not in g.ndata:
            print("we do not have train_mask in the graph, we create a dummy one")
            g.ndata['train_mask'] = torch.ones(g.num_nodes(), dtype=torch.int)
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
            
        model.eval()
        
           
       
        with torch.no_grad():  
            embeddings = model.inference(g, device, args.batch_size).cpu().numpy()
                ## Save embeddings
            os.makedirs(f"/mnt/reordered/embeddings", exist_ok=True)
            np.save(f"/mnt/reordered/embeddings/{name}.npy", embeddings)
        print("Embeddings computed.")
            
    print("loading graph again, as the other graph is bidirected for link prediction")
    g, num_classes = load_dgl_graph(args.dataset_to_partition)
        # for linkprediction where we do not have train_mask
    if 'train_mask' not in g.ndata:
        print("we do not have train_mask in the graph, we create a dummy one")
        g.ndata['train_mask'] = torch.ones(g.num_nodes(), dtype=torch.int)
    if 'val_mask' not in g.ndata:
        print("we do not have train_mask in the graph, we create a dummy one")
        g.ndata['val_mask'] = torch.ones(g.num_nodes(), dtype=torch.int)
    return embeddings,g

def compute_or_load_nodeprediction_embeddings(args):
    print("We perform node classification with GraphSage or GAT.")
    embeddings = None
    name = f"D{args.dataset}.E{args.epochs}.{args.model_name}.H{args.hidden_dims}.L{args.num_layers}.F{'-'.join(map(str, args.fanout))}"
    print(f"/mnt/reordered/embeddings/{name}.npy")
    print(f"{name}.npy")
        
        
    if f"{name}.npy" in os.listdir("/mnt/reordered/embeddings"):
        embeddings = np.load(f"/mnt/reordered/embeddings/{name}.npy")
        
    else: 
        print(f"Start loading dataset {args.dataset_to_partition} which should be partitioned.")

        print(f"Training in {args.mode} mode.")
            
        print(f"Start loading dataset {args.dataset_to_partition} which should be partitioned.")
        g, num_classes = load_dgl_graph(args.dataset_to_partition)
        print(g)
        print(f"Stop loading dataset {args.dataset_to_partition} which should be partitioned.")
                    # for linkprediction where we do not have train_mask
        if 'train_mask' not in g.ndata:
            print("we do not have train_mask in the graph, we create a dummy one")
            g.ndata['train_mask'] = torch.ones(g.num_nodes(), dtype=torch.int)    
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
        model.eval()
        with torch.no_grad():  
            embeddings = model.inference_sampling(model, g, device, args.batch_size,args.fanout).cpu().numpy()
            os.makedirs(f"/mnt/reordered/embeddings", exist_ok=True)
            np.save(f"/mnt/reordered/embeddings/{name}.npy", embeddings)
       
       
    g, num_classes = load_dgl_graph(args.dataset_to_partition)
    if 'train_mask' not in g.ndata:
        print("we do not have train_mask in the graph, we create a dummy one")
        g.ndata['train_mask'] = torch.ones(g.num_nodes(), dtype=torch.int)   
            
    if 'val_mask' not in g.ndata:
        print("we do not have train_mask in the graph, we create a dummy one")
        g.ndata['val_mask'] = torch.ones(g.num_nodes(), dtype=torch.int)
    return embeddings,g,name

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
    
    parser.add_argument(
        "--pca",
        type=int,
        help="the number of principal components to keep",
        required=False,
        default=-1
    )
     
    args = parser.parse_args()
    
    print(args)
    
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")
    
    
    embeddings_original = None
    embeddings = None
    name = None
    if args.model_name == "linkgraphsage":
        print("We perform link prediction with LinkGraphSage.")       
        embeddings, g = compute_or_load_linkgraphsage_embeddings(args)
        name = f"D{args.dataset}.E{args.epochs}.{args.model_name}.H{args.hidden_dims}.L{args.num_layers}.O{args.out_size}.F{'-'.join(map(str, args.fanout))}.TB{args.max_training_balance}.VB{args.max_vertex_balance}"


    if args.model_name in ["graphsage", "gat"]:
        print("We perform node classification with GraphSage or GAT.")
        embeddings, g, name = compute_or_load_nodeprediction_embeddings(args) 
        name = f"D{args.dataset}.E{args.epochs}.{args.model_name}.H{args.hidden_dims}.L{args.num_layers}.F{'-'.join(map(str, args.fanout))}.TB{args.max_training_balance}.VB{args.max_vertex_balance}"


    if not embeddings is None:
        t1 = time.time()
        embeddings_original = embeddings.copy()            
        print(f"Time to normalize embeddings: {time.time() - t1:.2f} seconds.")
        
    partitioner =  Partitioner(compute_metrics=False)
        
    

    # Experiment: we tune kmeans with different parameters
    if False:
        for p in args.num_parts:
            if args.partitioner == "k-means":
                print(f"We partition with K-Means into {p} partitions")    
                for niter in [8, 16, 64, 512]:
                    for max_points_per_centroid in [2**8, 2**10, 2**12, 2**14, 2**16]:
                        for nredo in [1, 5, 10, 15]:
                            
                            print(f"niter={niter}, max_points_per_centroid={max_points_per_centroid}, nredo={nredo}")              
                            embeddings = embeddings_original.copy()

                            vid2pid, metrics = kmeans(partitioner, g, embeddings, p, niter, nredo, max_points_per_centroid, args)
                            print(metrics)
                            store_metrics(args=args, metrics=metrics)
    
    
    
    # Experiment: we use PCA with different dimensions to reduce the embeddings before partitioning
    if False:
        for p in args.num_parts:
            if args.partitioner == "k-means":
                print(f"We partition with K-Means into {p} partitions")

                for pca in [-1, 4, 8, 16, 32]:
                    args.pca = pca
                    embeddings = embeddings_original.copy()
                    
                    if args.pca != -1:
                        if args.pca > embeddings.shape[1]:
                            print(f"Warning: PCA dimension {args.pca} is larger than the embedding dimension {embeddings.shape[1]}. Setting PCA to <= {embeddings.shape[1]}.")
                            continue
                        t1 = time.time()
                        embeddings = partitioner.PCA(embeddings, num_parts=p, d_out=args.pca)
                        duration_pca = time.time() - t1
                        print(f"Time to apply PCA: {duration_pca:.2f} seconds.")
                        
                    vid2pid, metrics = kmeans(partitioner=partitioner, graph=g, embeddings=embeddings, num_parts=p, niter=8, nredo=5, max_points_per_centroid=2**9, args=args)
                    
                    if args.pca != -1:
                        metrics["duration_pca"] = duration_pca
                        metrics["partitioning_time"] = metrics["partitioning_time"] + duration_pca     
                        
                    print(metrics)
                    store_metrics(args=args, metrics=metrics)   
        
    
    # This is the default partitioner, which is tuned
    if True:
        for p in args.num_parts:
            if args.partitioner == "k-means":
                embeddings = embeddings_original.copy()
                print(f"We partition with K-Means into {p} partitions")

                if args.pca != -1:
                    if args.pca > embeddings.shape[1]:
                        print(f"Warning: PCA dimension {args.pca} is larger than the embedding dimension {embeddings.shape[1]}. Setting PCA to <= {embeddings.shape[1]}.")
                        continue
                    t1 = time.time()
                    embeddings = partitioner.PCA(embeddings, num_parts=p, d_out=args.pca)
                    duration_pca = time.time() - t1
                    print(f"Time to apply PCA: {duration_pca:.2f} seconds.")
                    
                vid2pid, metrics = kmeans(partitioner=partitioner, graph=g, embeddings=embeddings, num_parts=p, niter=8, nredo=5, max_points_per_centroid=2**9, args=args)
                
                if args.pca != -1:
                    metrics["duration_pca"] = duration_pca
                    metrics["partitioning_time"] = metrics["partitioning_time"] + duration_pca     
                    
                print(metrics)
                store_metrics(args=args, metrics=metrics)  
                if args.not_store == False: 
                    print("store partition mapping")
                    fn = f"{args.dataset_to_partition}.{name}.P{p}.vid2pid"
                    store_partition_mapping(vid2pid=vid2pid, dir=PARTITIONED_GRAPHS, file=fn)
 
                
    
    if True:
        if args.partitioner == "feature-partitioning":
            g, num_classes = load_dgl_graph(args.dataset_to_partition)
            embeddings = g.ndata['feat'].numpy().copy()
            t1 = time.time()
            faiss.normalize_L2(embeddings)
            print(f"Time to normalize embeddings: {time.time() - t1:.2f} seconds.")
            for p in args.num_parts:
                print(f"We partition with K-Means into {p} partitions based on features")

                if args.pca != -1:
                    if args.pca > embeddings.shape[1]:
                        print(f"Warning: PCA dimension {args.pca} is larger than the embedding dimension {embeddings.shape[1]}. Setting PCA to <= {embeddings.shape[1]}.")
                        continue
                    t1 = time.time()
                    embeddings = partitioner.PCA(embeddings, num_parts=p, d_out=args.pca)
                    duration_pca = time.time() - t1
                    print(f"Time to apply PCA: {duration_pca:.2f} seconds.")
                    
                vid2pid, metrics = kmeans(partitioner=partitioner, graph=g, embeddings=embeddings, num_parts=p, niter=8, nredo=5, max_points_per_centroid=2**9, args=args)
                
                if args.pca != -1:
                    metrics["duration_pca"] = duration_pca
                    metrics["partitioning_time"] = metrics["partitioning_time"] + duration_pca     
                    
                print(metrics)
                store_metrics(args=args, metrics=metrics) 
                
                if args.not_store == False: 
                    print("store partition mapping")
                    fn = f"{args.dataset_to_partition}.feature-partitioning.P{p}.vid2pid"
                    store_partition_mapping(vid2pid=vid2pid, dir=PARTITIONED_GRAPHS, file=fn)
                    
                   
                
    
    exit()
    
    
    