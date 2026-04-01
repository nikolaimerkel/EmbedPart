import argparse
from datetime import datetime
import time
import pandas as pd
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


from training.eval import evaluate
from utils.graph_utils import load_dgl_graph
from models.model_factory import get_model
from models.utils import save_model, load_model

def train(args, device, g, model, num_classes):
    train_idx = torch.nonzero(g.ndata['train_mask'], as_tuple=False).squeeze().to(device)
    val_idx = torch.nonzero(g.ndata['val_mask'], as_tuple=False).squeeze().to(device)
    test_idx = torch.nonzero(g.ndata['test_mask'], as_tuple=False).squeeze().to(device)

    sampler = NeighborSampler(
        args.fanout,
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024*8,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    val_dataloader = DataLoader(
        g,
        val_idx,
        sampler,
        device=device,
        batch_size=1024*8,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
    
    
    # we have trained 0 epochs, meaning no training

    if args.no_checkpoint == "no":
        print("Saving initial model state")
       # save_model(model, opt, 0, args.model_name, args.dataset, args.hidden_dims, args.num_layers, args.fanout)


    metrics = []
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)            
            loss = F.cross_entropy(y_hat,torch.round(y).long())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader, num_classes)
        metrics.append({
            "epoch": epoch,
            "loss": total_loss / (it + 1),
            "accuracy": acc.item()
        })
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc.item()
            )
        )
        # we trained at least one epoch
        if args.no_checkpoint == "no":
            print(f"Saving model state for epoch {epoch + 1}")
      #      save_model(model, opt, epoch+1, args.model_name, args.dataset, args.hidden_dims, args.num_layers, args.fanout)
    
    pd.DataFrame(metrics).to_csv(f"results/gnn-partitioner-accuracy/{args.model_name}_{args.dataset}_{args.hidden_dims}_{args.num_layers}_{'_'.join(map(str, args.fanout))}_metrics.csv", index=False)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        default="mixed",
        choices=["cpu", "mixed", "puregpu"],
        help="Training mode. 'cpu' for CPU training, 'mixed' for CPU-GPU mixed training, "
        "'puregpu' for pure-GPU training.",
        required=True
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="graphsage",
        help="The model to train.",
        required=True
    )
     
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="The graph to train on.",
        required=True
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024*8,
        help="The batch size to use.",
        required=True
    )
    
    parser.add_argument(
        "--hidden_dims",
        type=int,
        default=64,
        help="The number of hidden dimensions.",
        required=True
    )
    
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="The number of layers in the model.",
        required=True
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="The number of epochs to train for.",
        required=True
    )
    
    parser.add_argument(
        '-fanout', 
        nargs='+', 
        type=int, 
        help="The fanout for all layers.",
        required=True
    )
    
    parser.add_argument(
        "--no_checkpoint",
        type=str,
        default="no",
        help="if yes, model will not be stored..",
    )
    
    
    
    args = parser.parse_args()
    print(args)
    
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")
    
    print("Loading graph")
    g, num_classes = load_dgl_graph(args.dataset)
    print(g)
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
   
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    train(args, device, g, model, num_classes)
    
    
   