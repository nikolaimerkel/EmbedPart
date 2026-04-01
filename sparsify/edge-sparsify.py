import argparse
import json
import dgl
from partitioner.helper import get_partition_metrics, get_balance
import numpy as np
import torch
from utils.graph_utils import load_dgl_graph

import argparse
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import SAGEConv

import argparse

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


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # three-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "mean"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "mean"))
        self.dropout = nn.Dropout(0.5)
        self.hid_size = hid_size
        self.out_size = out_size

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h

    def inference(self, g, device, batch_size):
        """Conduct layer-wise inference to get all the node embeddings."""
        feat = g.ndata["feat"]
        sampler = MultiLayerFullNeighborSampler(1, prefetch_node_feats=["feat"])
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to(g.device),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
        )
        buffer_device = torch.device("cpu")
        pin_memory = buffer_device != device

        for l, layer in enumerate(self.layers):
            y = torch.empty(
                g.num_nodes(),
                self.hid_size if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)  # len(blocks) = 1
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                    h = self.dropout(h)
                # by design, our output nodes are contiguous
                y[output_nodes[0] : output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y


def evaluate(model, graph, dataloader, num_classes):
    model.eval()
    ys = []
    y_hats = []
    for it, (input_nodes, output_nodes, blocks) in enumerate(dataloader):
        with torch.no_grad():
            x = blocks[0].srcdata["feat"]
            ys.append(blocks[-1].dstdata["label"])
            y_hats.append(model(blocks, x))
    return MF.accuracy(
        torch.cat(y_hats),
        torch.cat(ys),
        task="multiclass",
        num_classes=num_classes,
    )


def layerwise_infer(device, graph, nid, model, num_classes, batch_size):
    model.eval()
    with torch.no_grad():
        # pred lives on the "buffer_device" used by model.inference (often 'cpu')
        pred = model.inference(graph, device, batch_size)   # [N, C]

        # --- unify devices & types BEFORE indexing ---
        if isinstance(nid, torch.Tensor) is False:
            nid = torch.as_tensor(nid, device=pred.device)
        else:
            nid = nid.to(pred.device)

        labels = graph.ndata["label"]
        if labels.dtype != torch.long:
            labels = labels.long()
        labels = labels.to(pred.device)

        # index after everything is on the same device
        pred_sel = pred.index_select(0, nid)
        label_sel = labels.index_select(0, nid)

        return MF.accuracy(pred_sel, label_sel, task="multiclass", num_classes=num_classes)


def train(args, device, g, train_idx, val_idx, model, num_classes):
    # create sampler & dataloader
  #  train_idx = dataset.train_idx.to(device)
   # val_idx = dataset.val_idx.to(device)
    train_idx = train_idx.to(device)
    val_idx = val_idx.to(device)
    sampler = NeighborSampler(
        [10, 10, 10],  # fanout for [layer-0, layer-1, layer-2]
        prefetch_node_feats=["feat"],
        prefetch_labels=["label"],
    )
    use_uva = args.mode == "mixed"
    train_dataloader = DataLoader(
        g,
        train_idx,
        sampler,
        device=device,
        batch_size=1024,
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
        batch_size=1024,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        use_uva=use_uva,
    )

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for it, (input_nodes, output_nodes, blocks) in enumerate(
            train_dataloader
        ):
            x = blocks[0].srcdata["feat"]
            y = blocks[-1].dstdata["label"]
            y_hat = model(blocks, x)
            loss = F.cross_entropy(y_hat, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        acc = evaluate(model, g, val_dataloader, num_classes)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, total_loss / (it + 1), acc.item()
            )
        )


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-graph', 
        type=str, 
        help="Absolute path to the graph file.",
        required=True
    )
    parser.add_argument(
        '-vid2pid', 
        type=str, 
        help="Absolute path to the vid2pid file.",
        required=True
    )
    
    parser.add_argument(
        '-num_parts', 
        type=int, 
        help="The number of partitinos the graph was partitioned into.",
        required=True
    )
    
    parser.add_argument(
        '-partitioner', 
        type=str, 
        help="The partitioner used to partition the graph.",
        required=True
    )
    
    parser.add_argument(
        '-graph_name', 
        type=str, 
        help="The mane of the graph.",
        required=True
    )
    parser.add_argument(
        '-remove_frac', 
        type=float, 
        help="Fraction of remote edges to remove.",
        required=False,
        default=0.5
    )
    
    
    return parser

# python -m sparsify.edge-sparsify -graph /mnt/data/dgl/ogbn-arxiv.dgl -vid2pid /mnt/data/partitioned/ogbn-arxiv.metis.P4.vid2pid -num_parts 4 -partitioner metis -graph_name ogbn-arxiv

def start(args, g, num_classes):

    #with open(f"/mnt/data/gnn-partitioning/results/partitioning-metrics/cpp/{args.graph_name}.{args.partitioner}.P{args.num_parts}.json", "w") as f:
     #   f.write(json.dumps(metrics_clean, separators=(',', ':')) + '\n')
            
    args.mode = "puregpu"  # cpu, mixed, puregpu
    args.hidden = 256
    args.dropout = 0.5
    args.lr = 1e-2
    args.epochs = 20
    args.weight_decay = 5e-4
    args.agg = "mean"
    args.seed = 0
    
    if not torch.cuda.is_available():
        args.mode = "cpu"
    print(f"Training in {args.mode} mode.")

    # load and preprocess dataset
    print("Loading data")
  
    g = g.to("cuda" if args.mode == "puregpu" else "cpu")
    #num_classes = dataset.num_classes

    device = torch.device("cpu" if args.mode == "cpu" else "cuda")

    # create GraphSAGE model
    in_size = g.ndata["feat"].shape[1]
    out_size = num_classes
    model = SAGE(in_size, 256, out_size).to(device)

    # convert model and graph to bfloat16 if needed

    # model training
    print("Training...")
    train_idx = torch.nonzero(g.ndata["train_mask"], as_tuple=False).squeeze(1)
    val_idx = torch.nonzero(g.ndata["val_mask"], as_tuple=False).squeeze(1)
    train(args, device, g, train_idx, val_idx, model, num_classes)

    # test the model
    print("Testing...")
    test_idx = torch.nonzero(g.ndata["test_mask"], as_tuple=False).squeeze(1)

    acc = layerwise_infer(
        device, g, test_idx, model, num_classes, batch_size=4096
    )
    print("Test Accuracy {:.4f}".format(acc.item()))
    
    
    

if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    print(args)
    # python -m scripts.compute_partition_metrics_extern -graph /mnt/data/dgl/ogbn-arxiv.dgl -vid2pid /mnt/data/partitioned/ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L3.F15-10-5.P2.vid2pid -cpp_metrics /mnt/data/edgelists/ogbn-arxiv.directed.ldg.2.edgecut.partitioning.metrics.json -num_parts 2
              
  # load graph
    #g = dgl.load_graphs(args.graph)[0][0]
    g, num_classes = load_dgl_graph(args.graph)


    print("Original graph:", g)

    # load partition mapping
    v2p = np.loadtxt(args.vid2pid, dtype=int)

    # edges in eid order
    src, dst = g.edges(order="eid")
    src_parts = torch.from_numpy(v2p[src])
    dst_parts = torch.from_numpy(v2p[dst])

    # find remote edges
    remote_eids = torch.nonzero(src_parts != dst_parts, as_tuple=False).squeeze(1)
    num_remote = remote_eids.numel()
    num_remove = int(args.remove_frac * num_remote)

    # just randomly remove edges
#    remote_eids = torch.nonzero(src_parts == src_parts, as_tuple=False).squeeze(1)

    print(f"Remote edges: {num_remote}, removing {num_remove}")

    # sample random subset of remote edges to remove
    perm = torch.randperm(num_remote)

    edges_to_remove = remote_eids[perm[:num_remove]]



    # clone + remove edges → preserves features & masks
    g2 = g.clone()
    g2.remove_edges(edges_to_remove)
    metrics = get_partition_metrics(g, v2p, args.num_parts)
    metrics_sparsified = get_partition_metrics(g2, v2p, args.num_parts)
    
    print("Metrics original graph:", metrics)
    print("Metrics sparsified graph:", metrics_sparsified)

    print("\n")
    print("We start with the original graph which has #edges:", g.number_of_edges())
    start(args, g, num_classes)
    start(args, g, num_classes)
    
    
    print("\n")
    print("We continue with the sparsified graph which has #edges:", g2.number_of_edges())
    start(args, g2, num_classes)
    start(args, g2, num_classes)
    
    
    
    # .Dogbn-products.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1.

        
    # python -m sparsify.edge-sparsify -graph ogbn-products -vid2pid /mnt/data/partitioned/ogbn-products.Dogbn-products.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1.P4.vid2pid -num_parts 4 -partitioner metis -graph_name ogbn-products -remove_frac 0.9
    
    # python -m sparsify.edge-sparsify -graph ogbn-products -vid2pid /mnt/data/partitioned/ogbn-products.metis.P4.vid2pid -num_parts 4 -partitioner metis -graph_name ogbn-products -remove_frac 1
    
    # python -m sparsify.edge-sparsify -graph ogbn-arxiv -vid2pid /mnt/data/partitioned/ogbn-arxiv.metis.P4.vid2pid -num_parts 4 -partitioner metis -graph_name ogbn-arxiv -remove_frac 1
      
        
    
    
        
