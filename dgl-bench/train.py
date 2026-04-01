import argparse
import os
import time
import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

def train(graph,args):
    # Move graph to device
    device = torch.device("cpu")
    
    if args.infrastructure == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
        args.infrastructure = "cuda"
    else:
        args.infrastructure = "cpu"      
        
   
    print(f"Using device: {device}")
        
   # device = torch.device("cpu")
    graph = graph.to(device)

    # Node features and labels
    features = graph.ndata['feat']
    labels = graph.ndata['label']

    # Train/val/test masks
    train_mask = graph.ndata['train_mask']
    val_mask = graph.ndata['val_mask']
    test_mask = graph.ndata['test_mask']

    # Define a two-layer GCN
    from dgl.nn import GraphConv

    class GCN(nn.Module):
        def __init__(self, in_feats, hidden_feats, num_classes, num_layers):
            super(GCN, self).__init__()
            assert num_layers >= 2, "GCN requires at least 2 layers"

            self.layers = nn.ModuleList()
            self.layers.append(GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True))
            
            for _ in range(num_layers - 2):
                self.layers.append(GraphConv(hidden_feats, hidden_feats, allow_zero_in_degree=True))
            
            self.layers.append(GraphConv(hidden_feats, num_classes, allow_zero_in_degree=True))

        def forward(self, g, x):
            for layer in self.layers[:-1]:
                x = layer(g, x)
                x = F.relu(x)
            x = self.layers[-1](g, x)
            return x

    # Model setup
    model = GCN(in_feats=features.shape[1], hidden_feats=args.hidden_dim, num_classes=num_classes, num_layers=args.num_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.CrossEntropyLoss()

    epoch_times = []
    # Full-graph training loop
    for epoch in range(args.num_epochs):
        start_time = time.time()
        model.train()
        logits = model(graph, features)
        loss = loss_fn(logits[train_mask], labels[train_mask].long())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluation
       # model.eval()
       # with torch.no_grad():
        #    pred = logits.argmax(dim=1)
         #   acc = (pred[val_mask] == labels[val_mask]).float().mean()
        #if epoch % 10 == 0:
         #   print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {acc.item():.4f}")

        stop_time = time.time()
        epoch_times.append(stop_time - start_time)
        print(f"Epoch {epoch:03d} | Time: {stop_time - start_time:.4f} seconds")

    # Final test accuracy
    model.eval()
    pred = None
    with torch.no_grad():
        start_time_test = time.time()
        e = model(graph, features)
        pred = e.argmax(dim=1)
        test_acc = (pred[test_mask] == labels[test_mask]).float().mean()
        stop_time_test = time.time()
        print(f"Test Accuracy: {test_acc.item():.4f}")
        
        log_data = vars(args)
        log_data["test_accuracy"] = test_acc.item()
        log_data["epoch_times"] = epoch_times
        log_data["test_time"] = stop_time_test - start_time_test
        log_data["epoch_times_mean"] = np.mean(epoch_times)
        log_data["epoch_times_mean_after5"] = np.mean(epoch_times[5:])
        
        with open(f"{args.results_dir}", "w") as f:
            f.write(json.dumps(log_data, separators=(',', ':')) + '\n')
            
        print(f"log_data: {log_data}")
            
                
    return e 


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '-absolute_path_to_graph',  
        type=str, 
        help="The absolute path to the graph",
        required=True
    )
    
    parser.add_argument(
        '-infrastructure',  
        type=str, 
        help="cuda or cpu",
        default="cpu",
        required=False
    )
    
    parser.add_argument(
        '-num_epochs',  
        type=int, 
        help="Number of epochs",
        default=20,
        required=False
    )
    
    parser.add_argument(
        '-num_layers',  
        type=int, 
        help="Number of layers",
        default=2,
        required=False
    )
    
    parser.add_argument(
        '-hidden_dim',
        type=int,
        help="Hidden dimension",
        default=64,
        required=False
    )
    
    parser.add_argument(
        '-results_dir',
        type=str,
        help="Results directory",
        default="/mnt/data/gnn-partitioning/results/single-node-training/1.json",
        required=False
    )
    
    args = parser.parse_args()
    
    graphs, metadata = dgl.load_graphs(args.absolute_path_to_graph)
    print("metadata", metadata)

    # Retrieve the first (and in your case, only) graph
    g_reordered = graphs[0]

    # Retrieve num_classes from the metadata
    num_classes = metadata['num_classes'].item()
    
    print(f"Graph {args.absolute_path_to_graph} loaded", g_reordered)

    # Train the model
    train(g_reordered, args)
