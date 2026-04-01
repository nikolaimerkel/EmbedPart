from operator import index
import numpy as np
from partitioner.helper import get_partition_metrics, get_balance
import numpy as np
import math
from partitioner.Migration import Migration
from partitioner.HyperPlanes import HyperPlanes
import faiss
import torch
import dgl
import json
import os
import time
import torch.nn.functional as F

class Partitioner:
    
    
    def __init__(self, compute_metrics=False):
        """
        Initialize the Partitioner object.

        Args:
            compute_metrics (bool, optional): Flag indicating whether to compute metrics. Defaults to False.
        """
        self.compute_metrics = compute_metrics
            
    def metrics(self, graph, vid2pid, num_parts, strategy):  
        if self.compute_metrics:
            metrics = get_partition_metrics(graph, vid2pid, num_parts)
            metrics['strategy'] = strategy
            return metrics
        return {}
    
    def partition_metricsmetrics(self, graph, vid2pid, num_parts, strategy):  
        metrics = get_partition_metrics(graph, vid2pid, num_parts)
        metrics['strategy'] = strategy
        return metrics

              
    def metis_partitioning(self, graph, graph_name, num_parts, cache_disc=True):
        # Here we store the partitioned graph on disc.
        partition_file_name = f"output/{graph_name}/metis-{num_parts}"
        
        # Here we store the mapping of vertex to partition id.
        cache_file_name = f"cached_mappings/{graph_name}-metis-{num_parts}.npy"
        
        if cache_disc and os.path.exists(cache_file_name):
            print("We are reading vertex mapping from disc / cache.")
            vid2pid = np.load(cache_file_name)
            metrics = self.metrics(graph, vid2pid, num_parts, "metis")
            return vid2pid, metrics
        
        # Partition the graph using Metis and write partitioned files out.
        graph.ndata["orginal_id"] = torch.arange(0, graph.num_nodes())
      #  graph.ndata["balance"] = torch.ones(graph.num_nodes())
      #  dgl.distributed.partition_graph(
      #      graph,
      #      part_method="metis",
      #      graph_name=graph_name,
      #      num_parts=num_parts,
      #      out_path=partition_file_name,
      #      return_mapping=True,
      #      balance_ntypes=graph.ndata['balance'],
      #      #balance_edges=True
      #      )
        dgl.distributed.partition_graph(
            graph,
            part_method="metis",
            graph_name=graph_name,
            num_parts=num_parts,
            out_path=partition_file_name,
            return_mapping=True,
            balance_ntypes=graph.ndata['train_mask'],
            balance_edges=True
            )
            
    
            
        
        conf = f"{partition_file_name}/{graph_name}.json"
        with open(conf) as file:
            data = json.load(file)
            num_nodes = data["num_nodes"]
            num_parts = data["num_parts"]
            vid2pid = np.zeros(num_nodes, dtype=np.int64)
            for pid in range(num_parts):
                p = dgl.distributed.partition.load_partition(conf, part_id=pid)
                local_vertices = p[1]["_N/orginal_id"].numpy()
                for l in local_vertices:
                    vid2pid[l] = pid
            metrics = self.metrics(graph, vid2pid, num_parts, "metis")
            if cache_disc:
                np.save(cache_file_name, vid2pid)
            return vid2pid, metrics
    
    def random_partitioning(self, graph, graph_name, num_parts, cache_disc=True):
        # Here we store the partitioned graph on disc.
        partition_file_name = f"output/{graph_name}/random-{num_parts}"
        
        # Here we store the mapping of vertex to partition id.
        cache_file_name = f"cached_mappings/{graph_name}-random-{num_parts}.npy"
        
        if cache_disc and os.path.exists(cache_file_name):
            print("We are reading vertex mapping from disc / cache.")
            vid2pid = np.load(cache_file_name)
            metrics = self.metrics(graph, vid2pid, num_parts, "random")
            return vid2pid, metrics
        
        vid2pid =  np.random.randint(0, num_parts, size=graph.number_of_nodes())
        if cache_disc:
            np.save(cache_file_name, vid2pid)
            
        return vid2pid, self.metrics(graph, vid2pid, num_parts, "random_partitioning")
    
    def prob_partitioner(self, graph, num_parts, embeddings, rebalance=False):
        _, vid2pid = torch.max(embeddings, dim=1)
        
        if num_parts != embeddings.shape[1]:
            print("ERROR: \n Number of partitions does not match the number of columns in the embeddings.")
            return None
        
        # convert vid2pid to numpy array from pytorch tensor
        vid2pid = vid2pid.numpy()
        if rebalance:
            vid2pid = Migration.random_migration(vid2pid, num_parts)
            return vid2pid, self.metrics(graph, vid2pid, num_parts, "prop-balanced")
        return vid2pid, self.metrics(graph, vid2pid, num_parts, "prop")

    def fennel_partitioning(self, graph,graph_name, num_parts, cache_disc=True):
        cache_file_name = f"cached_mappings/{graph_name}-fennel-{num_parts}.npy"
        
        if cache_disc and os.path.exists(cache_file_name):
            print("We are reading vertex mapping from disc / cache.")
            vid2pid = np.load(cache_file_name)
            metrics = self.metrics(graph, vid2pid, num_parts, "fennel")
            return vid2pid, metrics
        
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        v = 1.1 
        gamma = 1.5
        alpha = (math.sqrt(num_parts) * num_edges) / ( math.pow(num_nodes, 3.0/2.0) )
        load_limit = v * (num_nodes / float(num_parts))
        #print(f"num_nodes : {num_nodes}, num_edges : {num_edges}, alpha : {alpha}, load_limit : {load_limit}, num_partitions : {num_partitions}, gamma : {gamma}, v : {v}")
        vid2pid = np.zeros(num_nodes, dtype=int)
        pid2load = np.zeros(num_parts)
        
        for node in range(num_nodes):
            if (node % 10000) == 0:
                print(f"compute node : {node}")
            pid2score = np.zeros(num_parts, dtype=float)
           
            nids = graph.successors(node)
            for nid in nids:
                
                # neigbor needs to be assigened to a partition else it would have partition 0 by default
                if nid < node:
                    pid2score[vid2pid[nid]] = pid2score[vid2pid[nid]] + 1
            
            balance_penalty = alpha * gamma * pid2load**(gamma - 1)
           # if (node > num_nodes - 100):
            #    print("current node", node)
             #   print("pid2score", pid2score)
            pid2score = pid2score - balance_penalty
            # Check if the load limit is reached
            pid2score[pid2load > load_limit] = -10000000
            #if (node > num_nodes - 100):
             #   print("pid2score", pid2score)
              #  print("nids", nids)
               # print("pid2load",pid2load)
            
            #best_pid = np.argmax(pid2score)
            best_pid = np.random.choice(np.flatnonzero(pid2score == pid2score.max()))
            
            vid2pid[node] = best_pid
            pid2load[best_pid] = pid2load[best_pid] + 1
            
        if cache_disc:
            np.save(cache_file_name, vid2pid)
        return vid2pid, self.metrics(graph, vid2pid, num_parts, "fennel")
    
    
    def fast_partitioner(self, graph, embeddings, num_parts, train_balance=1.05, vertex_balance=1.1):
        # Use already imported torch and numpy
        D = embeddings.shape[1]  # Dimension of embeddings
        K = num_parts  # number of partitions

        class FastPartitioner(torch.nn.Module):
            def __init__(self, dim, k):
                super().__init__()
                self.proj = torch.nn.Linear(dim, k, bias=False)
                torch.nn.init.normal_(self.proj.weight, std=0.1)  # Gaussian init

            def forward(self, x):
                return torch.argmax(self.proj(x), dim=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FastPartitioner(dim=D, k=K).to(device)
        model.eval()
        print("We are on device", device)
        batch_size = 10000000  # Adjust batch size as needed to fit GPU memory
        num_nodes = embeddings.shape[0]
        vid2pid = np.empty(num_nodes, dtype=np.int64)

        with torch.no_grad():
            for start in range(0, num_nodes, batch_size):
                end = min(start + batch_size, num_nodes)
                batch_embeddings = torch.from_numpy(embeddings[start:end]).to(device)
                batch_vid2pid = model(batch_embeddings).cpu().numpy()
                vid2pid[start:end] = batch_vid2pid

        print("vid2pid[:10] =", vid2pid[:10])  # Example output
       # vid2pid_migrated = Migration.migrate_training_balance_fast(vid2pid, num_parts, graph, train_balance, vertex_balance)
        vid2pid_migrated = vid2pid
        return vid2pid_migrated, self.metrics(graph, vid2pid_migrated, num_parts, "kmeans")
    
    
    def fast_partitioner1D(self, graph, embeddings, num_parts, train_balance=1.05, vertex_balance=1.1):
        D = embeddings.shape[1]
        K = num_parts

        class FastPartitioner1D(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.proj = torch.nn.Linear(dim, 1, bias=False)
                torch.nn.init.normal_(self.proj.weight, std=0.1)  # Random Gaussian vector

            def forward(self, x):
                return self.proj(x).squeeze(-1)  # [N, 1] -> [N]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FastPartitioner1D(dim=D).to(device)
        model.eval()

        num_nodes = embeddings.shape[0]
        projections = np.empty(num_nodes, dtype=np.float32)

        batch_size = 1000000
        with torch.no_grad():
            for start in range(0, num_nodes, batch_size):
                end = min(start + batch_size, num_nodes)
                batch = torch.from_numpy(embeddings[start:end]).to(device)
                projections[start:end] = model(batch).cpu().numpy()

        # Use quantiles to partition into K ranges
        thresholds = np.percentile(projections, q=np.linspace(0, 100, K + 1)[1:-1])  # K-1 cut points
        vid2pid = np.digitize(projections, thresholds, right=False)

        print("vid2pid[:10] =", vid2pid[:10])
        vid2pid_migrated = Migration.migrate_training_balance_fast(vid2pid, num_parts, graph, train_balance, vertex_balance)
        return vid2pid_migrated, self.metrics(graph, vid2pid_migrated, num_parts, "proj-1D-range")


    def fast_partitioner1D2(self, graph, embeddings, num_parts, train_balance=1.05, vertex_balance=1.1):
        D = embeddings.shape[1]
        K = num_parts

        class FastPartitioner1D(torch.nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.proj = torch.nn.Linear(dim, 1, bias=False)
                torch.nn.init.normal_(self.proj.weight, std=0.1)  # Random Gaussian vector

            def forward(self, x):
                return self.proj(x).squeeze(-1)  # [N, 1] -> [N]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = FastPartitioner1D(dim=D).to(device)
        model.eval()

        num_nodes = embeddings.shape[0]
        projections = np.empty(num_nodes, dtype=np.float32)

        batch_size = 1000000
        with torch.no_grad():
            for start in range(0, num_nodes, batch_size):
                end = min(start + batch_size, num_nodes)
                batch = torch.from_numpy(embeddings[start:end]).to(device)
                projections[start:end] = model(batch).cpu().numpy()

        # K-independent: sort the projections and assign partitions by slicing
        sorted_indices = np.argsort(projections)
        vid2pid = np.empty(num_nodes, dtype=np.int32)
        for i in range(K):
            start = i * (num_nodes // K)
            end = (i + 1) * (num_nodes // K) if i < K - 1 else num_nodes
            vid2pid[sorted_indices[start:end]] = i

        print("vid2pid[:10] =", vid2pid[:10])
        vid2pid_migrated = Migration.migrate_training_balance_fast(vid2pid, num_parts, graph, train_balance, vertex_balance)
        return vid2pid_migrated, self.metrics(graph, vid2pid_migrated, num_parts, "proj-1D-sorted")
    
    
    def PCA(self, embeddings, num_parts, d_out=8):
        d_in = embeddings.shape[1]   
        pca_matrix = faiss.PCAMatrix(d_in, d_out)
        pca_matrix.train(embeddings)
        pca_matrix = pca_matrix.apply_py(embeddings)
        faiss.normalize_L2(pca_matrix) 
        return pca_matrix
   
    
    def kmeans(self, graph, embeddings, num_parts, niter=8, verbose= True, spherical=True, nredo=5, train_balance=1.05, vertex_balance=1.1, max_points_per_centroid=2**9):
        train_mask = graph.ndata['train_mask'].cpu().numpy()  # Get boolean mask
        train_nids = np.nonzero(train_mask)[0]     
        
        print("Max points per centroid:", max_points_per_centroid)
        
        degrees = graph.in_degrees()[train_nids]
        _, topk_indices = torch.topk(degrees, max_points_per_centroid*num_parts, largest=True)
        train_nids = train_nids[topk_indices.numpy()]
        
      ## # Compute in-degrees of the training nodes
      ##  degrees = graph.in_degrees()[train_nids]
      ##  # Build a mask: keep only vertices with degree > 1
      ##  mask = degrees > 2
      ##  # Apply the mask to both degrees and node IDs
      ##  degrees = degrees[mask]
      ##  train_nids = train_nids[mask]
      ##  # Now pick the top-k among the filtered nodes
      ##  _, topk_indices = torch.topk(-degrees, max_points_per_centroid * num_parts, largest=True)
      ##  train_nids = train_nids[topk_indices]
        
        
        
        # Get indices of training nodes
        train_embeddings = embeddings[train_nids]         # Select only training embeddings
        

        t1 = time.time()
        kmeans = faiss.Kmeans(train_embeddings.shape[1], num_parts, seed=42, niter=niter, verbose= verbose, spherical=spherical, nredo=nredo, max_points_per_centroid=max_points_per_centroid)
        kmeans.train(train_embeddings)
        t2 = time.time()
        duration_build = t2 - t1
        print(f"KMeans training completed in {duration_build:.2f} seconds.")

        t1 = time.time()
        D, I = kmeans.index.search(embeddings, 1)
        vid2pid = I.flatten()
        t2 = time.time()
        duration_predict = t2 - t1
        print(f"KMeans prediction completed in {duration_predict:.2f} seconds.")
        
        t1 = time.time()
        vid2pid_migrated =  Migration.migrate_training_balance_fast(vid2pid, num_parts, graph, train_balance, vertex_balance, distances=torch.from_numpy(D))
        t2 = time.time()
        duration_migration = t2 - t1
        print(f" Migration completed in {duration_migration:.2f} seconds.")
        
        metrics = {}
        metrics['duration_build'] = duration_build
        metrics['duration_predict'] = duration_predict
        metrics['duration_migration'] = duration_migration
        metrics['strategy'] = "kmeans"
        
        return vid2pid_migrated, metrics
    
    
