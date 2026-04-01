import argparse
import socket
import time
from contextlib import contextmanager
import os
# Force DGL and PyTorch to use the correct interface (enp66s0f0)
os.environ["GLOO_SOCKET_IFNAME"] = "enp66s0f0"
os.environ["TP_SOCKET_IFNAME"] = "enp66s0f0"
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
import dgl
import dgl.nn.pytorch as dglnn
from dgl.nn import EdgePredictor
import json
import os 


def get_directory(args):
    return f"{args.results_dir}/{args.graph_name}.{args.partitioner}.{args.num_parts}.{args.model}.{args.num_hidden}.{args.num_layers}.{args.batch_size}.{args.force_even}"
    
SAGE = "linkgraphsage"
GAT = "linkgat"
MODELS = [GAT, SAGE]



class DistGAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout, num_heads):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.num_heads = num_heads
        self.activation = activation
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList()

        # Input layer
        self.layers.append(dglnn.GATConv(
            in_feats, n_hidden, num_heads, feat_drop=dropout, attn_drop=dropout, activation=activation, allow_zero_in_degree=True
        ))

        # Hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.GATConv(
                n_hidden * num_heads, n_hidden, num_heads, feat_drop=dropout, attn_drop=dropout, activation=activation, allow_zero_in_degree=True
            ))

        # Output layer (no activation)
        self.layers.append(dglnn.GATConv(
            n_hidden * num_heads, n_classes, 1, feat_drop=dropout, attn_drop=dropout, activation=None, allow_zero_in_degree=True
        ))

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            # If not the last layer and num_heads > 1, flatten the output
            if i != len(self.layers) - 1:
                h = h.flatten(1)
                h = self.dropout(h)
            else:
                h = h.squeeze(1)  # output layer has only 1 head
        return h

    def inference(self, g, x, batch_size, device):
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )
        y = dgl.distributed.DistTensor(
            (g.num_nodes(), self.n_hidden * self.num_heads),
            th.float32,
            "h",
            persistent=True,
        )
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.n_classes),
                    th.float32,
                    "h_last",
                    persistent=True,
                )
            print(f"|V|={g.num_nodes()}, eval batch size: {batch_size}")

            sampler = dgl.dataloading.NeighborSampler([-1])
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(self.layers) - 1:
                    h = h.flatten(1)
                    h = self.dropout(h)
                else:
                    h = h.squeeze(1)  # final head
                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

    @contextmanager
    def join(self):
        yield
        
        

class DistSAGE(nn.Module):
    def __init__(
        self, in_feats, n_hidden, n_classes, n_layers, activation, dropout
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, "mean"))
        for i in range(1, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, "mean"))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, batch_size, device):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without
        neighbor sampling).

        g : the entire graph.
        x : the input of entire node set.

        Distributed layer-wise inference.
        """
        # During inference with sampling, multi-layer blocks are very
        # inefficient because lots of computations in the first few layers
        # are repeated. Therefore, we compute the representation of all nodes
        # layer by layer.  The nodes on each layer are of course splitted in
        # batches.
        # TODO: can we standardize this?
        nodes = dgl.distributed.node_split(
            np.arange(g.num_nodes()),
            g.get_partition_book(),
            force_even=True,
        )
        y = dgl.distributed.DistTensor(
            (g.num_nodes(), self.n_hidden),
            th.float32,
            "h",
            persistent=True,
        )
        for i, layer in enumerate(self.layers):
            if i == len(self.layers) - 1:
                y = dgl.distributed.DistTensor(
                    (g.num_nodes(), self.n_classes),
                    th.float32,
                    "h_last",
                    persistent=True,
                )
            print(
                f"|V|={g.num_nodes()}, eval batch size: {batch_size}"
            )

            sampler = dgl.dataloading.NeighborSampler([-1])
            dataloader = dgl.dataloading.DistNodeDataLoader(
                g,
                nodes,
                sampler,
                batch_size=batch_size,
                shuffle=False,
                drop_last=False,
            )

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0].to(device)
                h = x[input_nodes].to(device)
                h_dst = h[: block.number_of_dst_nodes()]
                h = layer(block, (h, h_dst))
                if i != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
            g.barrier()
        return y

    @contextmanager
    def join(self):
        """dummy join for standalone"""
        yield


def compute_acc(pred, labels):
    """
    Compute the accuracy of prediction given the labels.
    """
    labels = labels.long()
    return (th.argmax(pred, dim=1) == labels).float().sum() / len(pred)


def evaluate(model, g, inputs, labels, val_nid, test_nid, batch_size, device):
    """
    Evaluate the model on the validation set specified by ``val_nid``.
    g : The entire graph.
    inputs : The features of all the nodes.
    labels : The labels of all the nodes.
    val_nid : the node Ids for validation.
    batch_size : Number of nodes to compute at the same time.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        pred = model.inference(g, inputs, batch_size, device)
    model.train()
    return compute_acc(pred[val_nid], labels[val_nid]), compute_acc(
        pred[test_nid], labels[test_nid]
    )

def load_subtensor(g, seeds, input_nodes, device, load_feat=True):
    """
    Copys features and labels of a set of nodes onto GPU.
    """
    batch_inputs = (
        g.ndata["feat"][input_nodes].to(device) if load_feat else None
    )
    batch_labels = g.ndata["label"][seeds].to(device)
    return batch_inputs, batch_labels



def run(args, device, data):
    # Unpack data
    in_feats, n_classes, g = data
    print(data)
    train_eids = dgl.distributed.edge_split(th.ones((g.number_of_edges(),), dtype=th.bool), g.get_partition_book(), force_even=args.force_even)
    print(train_eids.shape, "train_eids shape")
    print(len(train_eids), "len train_eids")
    sampler = dgl.dataloading.MultiLayerNeighborSampler([int(fanout) for fanout in args.fan_out.split(",")])
    neg_sampler = dgl.dataloading.negative_sampler.Uniform(1)
    train_dataloader = dgl.dataloading.DistEdgeDataLoader(
        g, train_eids, sampler, negative_sampler=neg_sampler, batch_size=args.batch_size,
        shuffle=True, drop_last=False)

    shuffle = True

    
    model = None
    if args.model == SAGE:
        model = DistSAGE(
            in_feats,
            args.num_hidden,
            n_classes,
            args.num_layers,
            F.relu,
            args.dropout,
        )
    elif args.model == GAT:  
        model = DistGAT(
            in_feats,
            args.num_hidden,
            n_classes,
            args.num_layers,
            F.relu,
            args.dropout,
            8
        )
    else:
        raise ValueError("Unknown model type: {}".format(args.model))

    
            
    model = model.to(device)
    if not args.standalone:
        if args.num_gpus == -1:
            model = th.nn.parallel.DistributedDataParallel(model)
        else:
            model = th.nn.parallel.DistributedDataParallel(
                model, device_ids=[device], output_device=device
            )
    loss_fcn = nn.CrossEntropyLoss()
    loss_fcn = loss_fcn.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)


    predictor = EdgePredictor('dot')



    num_partitions = args.num_parts
    num_edges = g.number_of_edges()
    batch_size = args.batch_size
    
    num_steps = num_edges // batch_size // num_partitions
    print("we fill need to do", num_steps, "steps")
    
    
    # Training loop
    epoch = 0
    directory = get_directory(args)   
   # for epoch in range(args.num_epochs):
   
   
    step_time = []
    START = time.time()
    for epoch in range(args.num_epochs):
        print("Epoch", epoch)
        start = time.time()
        with model.join():
            for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(train_dataloader):
                if time.time() - START > 60*15:  
                    print("Stopping after 15 min")
                    break
                blocks = [block.to(device) for block in blocks]
                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                batch_inputs = g.ndata["feat"][input_nodes].to(device)
                batch_pred = model(blocks, batch_inputs)
                pos_features = batch_pred
                pos_graph.ndata['h'] = batch_pred
                pos_src, pos_dst = pos_graph.edges()
                pos_score = predictor(pos_features[pos_src], pos_features[pos_dst])
                neg_features = batch_pred
                neg_graph.ndata['h'] = batch_pred
                neg_src, neg_dst = neg_graph.edges()
                neg_score = predictor(neg_features[neg_src], neg_features[neg_dst])
                score = th.cat([pos_score, neg_score])
                labels = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)])
                loss = F.binary_cross_entropy_with_logits(score, labels)
                #print("Loss", loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                current_step_time = time.time() - start
                print("current_step_time", current_step_time)
                step_time.append(current_step_time)
                start = time.time()

      
        print(
            "Part {},  Step time {:.4f}".format(
                g.rank(),
                np.mean(step_time) if step_time else 0,
            
            )
        )
        
        part = g.rank()
        mean_step_time = np.mean(step_time[2:]) if step_time else 0        

        # Create dictionary
        log_data = {
            "part": part,
            "step_time": round(mean_step_time, 4)
        }
        
        
        # graph, partitioner, num parts 
        
        with open(f"{directory}/{part}.json", "a") as f:
            f.write(json.dumps(log_data, separators=(',', ':')) + '\n')
            
            
        
        epoch += 1


def main(args):
    
    

    print(socket.gethostname(), "Initializing DGL dist")
    dgl.distributed.initialize(args.ip_config, net_type=args.net_type)
    if not args.standalone:
        print(socket.gethostname(), "Initializing DGL process group")
        th.distributed.init_process_group(backend=args.backend)
    print(socket.gethostname(), "Initializing DistGraph")
    g = dgl.distributed.DistGraph(
            args.graph_name,
            part_config=args.part_config
        )
    print(socket.gethostname(), "rank:", g.rank())
    print(socket.gethostname(), " afterInitializing DistGraph")
    print("\n", g.ndata, "\n")

    pb = g.get_partition_book()
    
    local_nid = pb.partid2nids(pb.partid).detach().numpy()
    
    
    
    directory = get_directory(args)
    
        
    if g.rank() == 0:
        print("create directory for results")
        
        os.makedirs(directory, exist_ok=True)
        
        log_data = {
            "graph_name": args.graph_name,
            "partitioner": args.partitioner,
            "num_parts": args.num_parts,
            "num_hidden": args.num_hidden,
            "num_layers": args.num_layers,
            "batch_size": args.batch_size,
            "model": args.model,
            "force_even": args.force_even,
        }
        
        with open(f"{directory}/config.json", "a") as f:
            f.write(json.dumps(log_data, separators=(',', ':')) + '\n')
            # Create dictionary
    log_data = {
        "part": g.rank(),
        "num_nodes_pid": len(local_nid),
        "force_even": args.force_even
    }

    
    # graph, partitioner, num parts 
    
    
    while not os.path.exists(directory):
        print("sleeping for directory to be created")
        time.sleep(1)
    
    with open(f"{directory}/ids.{g.rank()}.json", "a") as f:
        f.write(json.dumps(log_data, separators=(',', ':')) + '\n')
        


    del local_nid
    if args.num_gpus == -1:
        device = th.device("cpu")
    else:
        dev_id = g.rank() % args.num_gpus
        device = th.device("cuda:" + str(dev_id))
    n_classes = args.n_classes
    
    ca = th.cuda.is_available()

    
    print("\n\n\n")
    print("Device", device, ca)
    print("\n\n\n")
    
    #if n_classes == 0:
    
    n_classes = 8
    print("#labels:", n_classes)

    # Pack data
    in_feats = g.ndata["feat"].shape[1]
    data = in_feats, n_classes, g
    run(args, device, data)
    print("parent ends")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument("--graph_name", type=str, help="graph name")
    parser.add_argument("--id", type=int, help="the partition id")
    parser.add_argument(
        "--ip_config", type=str, help="The file for IP configuration"
    )
    parser.add_argument(
        "--part_config", type=str, help="The path to the partition config file"
    )
    parser.add_argument(
        "--n_classes", type=int, default=0, help="the number of classes"
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="gloo",
        help="pytorch distributed backend",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=-1,
        help="the number of GPU device. Use -1 for CPU training",
    )
    parser.add_argument("--model", type=str, default=SAGE)
    parser.add_argument("--force_even", type=bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_hidden", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--fan_out", type=str, default="10,25")
    #parser.add_argument("--fan_out", type=str, default="1,1")
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--batch_size_eval", type=int, default=100000)
    parser.add_argument("--log_every", type=int, default=20)
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument(
        "--local-rank", type=int, help="get rank of the process"
    )
    parser.add_argument(
        "--standalone", action="store_true", help="run in the standalone mode"
    )
    parser.add_argument(
        "--pad-data",
        default=False,
        action="store_true",
        help="Pad train nid to the same length across machine, to ensure num "
             "of batches to be the same.",
    )
    parser.add_argument(
        "--net_type",
        type=str,
        default="socket",
        help="backend net type, 'socket' or 'tensorpipe'",
    )
    
    parser.add_argument(
        "--results_dir",
        type=str,
        help="where to save the results per worker",
        required=True
    )
    
    parser.add_argument(
        "--partitioner",
        type=str,
        help="the used partitioner",
        required=True
    )
    
    parser.add_argument(
        "--num_parts",
        type=int,
        help="the used number of partitions",
        required=True
    )
    
    
    args = parser.parse_args()

    print(args)
    
    print("GLOO_SOCKET_IFNAME", os.environ.get("GLOO_SOCKET_IFNAME"))
    print("TP_SOCKET_IFNAME",os.environ.get("TP_SOCKET_IFNAME"))

    main(args)