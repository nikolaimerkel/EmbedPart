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
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator

import dgl.function as fn

class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']
    
class LinkGraphSage(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.out_size = out_size
        self.hid_size = hid_size
        if n_layers == 1:
            self.layers.append(dglnn.SAGEConv(in_size, out_size, 'mean'))
        else: 
            self.layers.append(dglnn.SAGEConv(in_size, hid_size, 'mean'))
            for _ in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(hid_size, hid_size, 'mean'))
            self.layers.append(dglnn.SAGEConv(hid_size, out_size, 'mean'))
            
        self.pred = DotProductPredictor()

    def forward(self, pair_graph, neg_pair_graph, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = F.relu(h) 
        return self.pred(pair_graph, h), self.pred(neg_pair_graph, h)
    
    def inference(self, g, device, batch_size):
        """Layer-wise inference algorithm to compute GNN node embeddings."""
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
            y = None
            if l == len(self.layers) - 1:
                # For the last layer, we do not need to apply ReLU activation
                # and dropout.
                y = torch.empty(
                    g.num_nodes(),
                    self.out_size,
                    device=buffer_device,
                    pin_memory=pin_memory,
                )
            else:
                y = torch.empty(
                    g.num_nodes(),
                    self.hid_size,
                    device=buffer_device,
                    pin_memory=pin_memory,
                )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(
                dataloader, desc="Inference"
            ):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = F.relu(h)
                y[output_nodes] = h.to(buffer_device)
            feat = y
        return y