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


class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, n_layers, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(0.5)
        self.out_size = out_size
        self.hid_size = hid_size
        self.num_heads = num_heads

        if n_layers == 1:
            self.layers.append(dglnn.GATConv(in_size, out_size, num_heads,  allow_zero_in_degree=True))
        else:
            # Input layer
            self.layers.append(dglnn.GATConv(in_size, hid_size, num_heads, allow_zero_in_degree=True))
            # Hidden layers
            for _ in range(1, n_layers - 1):
                self.layers.append(dglnn.GATConv(hid_size * num_heads, hid_size, num_heads, allow_zero_in_degree=True))
            # Output layer
            self.layers.append(dglnn.GATConv(hid_size * num_heads, out_size, 1, allow_zero_in_degree=True))  # 1 head for output

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            # For all layers except last, flatten head outputs and apply relu/dropout
            if l != len(self.layers) - 1:
                # h shape: (N, num_heads, out_feats)
                h = h.flatten(1)
                h = F.relu(h)
                h = self.dropout(h)
            else:
                # Last layer: if num_heads == 1, squeeze head dim
                if h.shape[1] == 1:
                    h = h.squeeze(1)
        return h

    # The rest remains exactly as in your original class (inference_sampling and inference)
    # Copy from your code above without changes

    def inference_sampling(self, model, g, device, batch_size, fanout):
        sampler = NeighborSampler(
            fanout,  # fanout for [layer-0, layer-1, layer-2]
            prefetch_node_feats=["feat"],
            prefetch_labels=["label"],
        )
        dataloader = DataLoader(
            g,
            torch.arange(g.num_nodes()).to("cuda"),
            sampler,
            device=device,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            use_uva=True
        )
        step = 0
        y_p = torch.zeros(g.num_nodes(), self.out_size, device="cpu")
        for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
            if step % 1000 == 0:
                print(step)
            step += 1
            x = blocks[0].srcdata["feat"]
            y_hat = model(blocks, x)
            y_p[output_nodes] = y_hat.cpu()
        return y_p

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
                self.hid_size * self.num_heads if l != len(self.layers) - 1 else self.out_size,
                dtype=feat.dtype,
                device=buffer_device,
                pin_memory=pin_memory,
            )
            feat = feat.to(device)
            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                x = feat[input_nodes]
                h = layer(blocks[0], x)
                if l != len(self.layers) - 1:
                    h = h.flatten(1)
                    h = F.relu(h)
                    h = self.dropout(h)
                else:
                    if h.shape[1] == 1:
                        h = h.squeeze(1)
                y[output_nodes[0]: output_nodes[-1] + 1] = h.to(buffer_device)
            feat = y
        return y