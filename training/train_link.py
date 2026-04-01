import argparse

import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from dgl.dataloading import (
    as_edge_prediction_sampler,
    DataLoader,
    MultiLayerFullNeighborSampler,
    negative_sampler,
    NeighborSampler,
)
import time
from utils.graph_utils import load_dgl_graph
from models.model_factory import get_model
from models.utils import save_model, load_model
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from ogb.nodeproppred import DglNodePropPredDataset
from partitioner.Partitioner import *
from models.model_factory import get_model
import json
import numpy as np

from utils.graph_utils import load_dgl_graph
from models.model_factory import get_model
from models.utils import save_model, load_model

from configs.config import PARTITIONED_GRAPHS
import dgl.function as fn

def to_bidirected_with_reverse_mapping(g):
    """Makes a graph bidirectional, and returns a mapping array ``mapping`` where ``mapping[i]``
    is the reverse edge of edge ID ``i``. Does not work with graphs that have self-loops.
    """
    g_simple, mapping = dgl.to_simple(
        dgl.add_reverse_edges(g), return_counts="count", writeback_mapping=True
    )
    c = g_simple.edata["count"]
    num_edges = g.num_edges()
    mapping_offset = torch.zeros(
        g_simple.num_edges() + 1, dtype=g_simple.idtype
    )
    mapping_offset[1:] = c.cumsum(0)
    idx = mapping.argsort()
    idx_uniq = idx[mapping_offset[:-1]]
    reverse_idx = torch.where(
        idx_uniq >= num_edges, idx_uniq - num_edges, idx_uniq + num_edges
    )
    reverse_mapping = mapping[reverse_idx]
    # sanity check
    src1, dst1 = g_simple.edges()
    src2, dst2 = g_simple.find_edges(reverse_mapping)
    assert torch.equal(src1, dst2)
    assert torch.equal(src2, dst1)
    return g_simple, reverse_mapping

def train(args, device, g, reverse_eids, seed_edges, model):
    # create sampler & dataloader
    sampler = NeighborSampler(args.fanout, prefetch_node_feats=["feat"])
    sampler = as_edge_prediction_sampler(
        sampler,
        exclude="reverse_id",
        reverse_eids=reverse_eids,
        negative_sampler=negative_sampler.Uniform(1),
    )
    use_uva = args.mode == "mixed"
    dataloader = DataLoader(
        g,
        seed_edges,
        sampler,
        device=device,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.0005)
    save_model(model, opt, 0, args.model_name, args.dataset, args.hidden_dims, args.num_layers, args.fanout, args.out_size)

    for epoch in range(args.epochs):
        time_start = time.time()
        model.train()
        total_loss = 0
        for it, (input_nodes, pair_graph, neg_pair_graph, blocks) in enumerate(
            dataloader
        ):
            x = blocks[0].srcdata["feat"]
            pos_score, neg_score = model(pair_graph, neg_pair_graph, blocks, x)
            score = torch.cat([pos_score, neg_score])
            pos_label = torch.ones_like(pos_score)
            neg_label = torch.zeros_like(neg_score)
            labels = torch.cat([pos_label, neg_label])
            loss = F.binary_cross_entropy_with_logits(score, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            if (it + 1) == 1000:
                break
        print("Epoch {:05d} | Loss {:.4f}".format(epoch, total_loss / (it + 1)))
        save_model(model, opt, epoch+1, args.model_name, args.dataset, args.hidden_dims, args.num_layers, args.fanout,args.out_size)
        time_end = time.time()
        print(f"Epoch {epoch+1} took {time_end - time_start:.2f} seconds")

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
        "--out_size",
        type=int,
        default=16,
        help="The output size.",
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
    args = parser.parse_args()
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # load and preprocess dataset
    print("Loading data")
   # dataset = DglLinkPropPredDataset("ogbl-citation2")
   # g = dataset[0]
    
        
   ## from dgl.data import CoraGraphDataset
    ##g = CoraGraphDataset()[0]
    
    g, num_classes = load_dgl_graph(args.dataset)
    g = dgl.remove_self_loop(g)
    print(g)
    
    print(g)
    
    device = torch.device("cpu" if args.mode == "cpu" else "cuda")
    
    g, reverse_eids = to_bidirected_with_reverse_mapping(g)
    reverse_eids = reverse_eids.to(device)
    seed_edges = torch.arange(g.num_edges()).to(device)
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
  #  edge_split = dataset.get_edge_split()

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    model = get_model(
        args.model_name, 
        g.ndata["feat"].shape[1], 
        args.hidden_dims, 
        args.out_size, 
        args.num_layers
    )
    model.to(device)

    print("Training...")
    train(args, device, g, reverse_eids, seed_edges, model)
    