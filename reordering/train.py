
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import CoraGraphDataset
from utils.graph_utils import load_dgl_graph
import time
import faiss
import numpy as np
import os
import argparse
from datetime import datetime
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv

class GCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super().__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats, allow_zero_in_degree=True)
        self.conv2 = GraphConv(hidden_feats, num_classes, allow_zero_in_degree=True)

    def forward(self, g, x):
        x = self.conv1(g, x)
        x = F.relu(x)
        x = self.conv2(g, x)              # raw logits (no softmax)
        return x

@torch.no_grad()
def accuracy(logits, labels):
    pred = logits.argmax(dim=1)
    return (pred == labels).float().mean()

def train(graph, num_epochs, hidden_dim, num_classes, device, lr=1e-3):
    graph = graph.to(device)

    # Node features and labels
    features = graph.ndata['feat'].to(device)
    labels = graph.ndata['label'].long().to(device)

    # Train/val/test masks (ensure boolean)
    train_mask = graph.ndata['train_mask'].to(device).bool()
    val_mask   = graph.ndata['val_mask'].to(device).bool()
    test_mask  = graph.ndata['test_mask'].to(device).bool()

    # Model setup
    model = GCN(in_feats=features.shape[1], hidden_feats=hidden_dim, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    metrics = {"epoch_times": []}

    for epoch in range(num_epochs):
        t0 = time.time()

        # ---- Train ----
        model.train()
        logits = model(graph, features)
        loss = loss_fn(logits[train_mask], labels[train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ---- Validate (recompute with updated weights) ----
      #  model.eval()
       # with torch.no_grad():
        #    val_logits = model(graph, features)
         #   val_acc = accuracy(val_logits[val_mask], labels[val_mask])

        # bookkeeping
        metrics["epoch_times"].append(time.time() - t0)
        #metrics["train_loss"].append(loss.item())
        #metrics["val_acc"].append(val_acc.item())

       # if (epoch % log_every) == 0 or epoch == num_epochs - 1:
        #    print(f"Epoch {epoch:03d} | Loss: {loss.item():.4f} | Val Acc: {val_acc.item():.4f} | Time: {metrics['epoch_times'][-1]:.3f}s")

        
    
    # ---- Final test ----
    t_start = time.time()
    model.eval()
    with torch.no_grad():
        test_logits = model(graph, features)
        test_acc = accuracy(test_logits[test_mask], labels[test_mask]).item()
    metrics["test_time"] = time.time() - t_start
    metrics["test_acc"] = test_acc
    print(f"Test Accuracy: {test_acc:.4f}")

    return metrics

def parser():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--absolute_path_dgl_graph',
        type=str,
        help="The absolute path to the DGL graph",
        required=True
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        help="The directory to save the results",
        required=True
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        help="cpu or cuda",
        default="cpu"
    )

    return parser


if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.mode == "cpu":
        device = torch.device("cpu")
    
    if "ogbn-papers100M" in args.absolute_path_dgl_graph:
        device = torch.device("cpu")
        print("Using CPU for ogbn-papers100M as GPU may not have enough memory")
        
    print(f"we train on the device: ", device)
    
    graphs, label_dict = dgl.load_graphs(args.absolute_path_dgl_graph)

    g_reordered = graphs[0]

    num_classes = label_dict['num_classes'].item()

    metrics = train(g_reordered, num_epochs=10, hidden_dim=64, num_classes=num_classes, device=device, lr=1e-3)

    # add args dict to metrics
    metrics["absolute_path_dgl_graph"] = args.absolute_path_dgl_graph
    metrics["results_dir"] = args.results_dir
    metrics["mode"] = args.mode
    metrics["epoch_times_mean_after5"] = np.mean(metrics["epoch_times"][5:])
    metrics["epoch_times_max"] = np.max(metrics["epoch_times"])
    metrics["epoch_times_min"] = np.min(metrics["epoch_times"])
    
    
    m = {key: int(value) if isinstance(value, np.integer) else value for key, value in metrics.items()}
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")
    os.makedirs(args.results_dir, exist_ok=True)
    with open(f"{args.results_dir}/{timestamp}.json", "w") as f:
        json.dump(m, f)  
        
    print(metrics)