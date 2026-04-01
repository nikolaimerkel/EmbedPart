import numpy as np
import torch


def get_partition_metrics(g, v2p, num_parts):
    """
    Calculates various metrics related to graph partitioning.

    Args:
        g (Graph): The input graph.
        v2p (numpy.ndarray): Array mapping node IDs to partition IDs.
        num_parts (int): The total number of partitions.

    Returns:
        dict: A dictionary containing the following metrics:
            - num_parts (int): The total number of partitions.
            - num_edges (int): The total number of edges in the graph.
            - local_edges (int): The number of edges within the same partition.
            - cut_edges (int): The number of edges crossing different partitions.
            - edge_cut (float): The ratio of cut edges to total edges.
            - balance (float): The balance of partition sizes.

    """
    src_ids, dst_ids = g.edges()
    num_nodes = g.number_of_nodes()
    print(f"g.number_of_nodes() = {num_nodes}")
    print(f"max src_ids = {src_ids.max()}")
    print(f"max dst_ids = {dst_ids.max()}")

    src_ids_mapped = v2p[src_ids]
    dst_ids_mapped = v2p[dst_ids]
    local_edges = np.sum(src_ids_mapped == dst_ids_mapped)
    num_edges = len(src_ids_mapped)
    
    train_mask = g.ndata['train_mask']
    mask_edges = train_mask[dst_ids]     # bool mask over edges
    src_ids_train = src_ids[mask_edges]
    dst_ids_train = dst_ids[mask_edges]

    local_train_edges = np.sum(v2p[src_ids_train] == v2p[dst_ids_train])
    remote_train_edges = np.sum(v2p[src_ids_train] != v2p[dst_ids_train])

    metrics = {}
    
    for mask in ["train", "val", "test"]:
        if f"{mask}_mask" in g.ndata:
            mask_data = g.ndata[f"{mask}_mask"].cpu().numpy()
            nids = np.nonzero(mask_data)[0]
            v2p_mask = v2p[nids]    
            _, counts = np.unique(v2p_mask, return_counts=True)
            metrics[f'{mask}_balance'] = np.max(counts) / (np.sum(mask_data) / num_parts)
    
    
    unique_bins, counts = np.unique(v2p, return_counts=True)               
    
    # Compute degrees
    out_degrees = np.bincount(src_ids, minlength=num_nodes)
    in_degrees = np.bincount(dst_ids, minlength=num_nodes)
    total_degrees = out_degrees + in_degrees

    # Group degrees by partition using np.bincount
    edge_sums = np.bincount(v2p, weights=total_degrees, minlength=num_parts)
    in_edge_sums = np.bincount(v2p, weights=in_degrees, minlength=num_parts)
    out_edge_sums = np.bincount(v2p, weights=out_degrees, minlength=num_parts)

    # Add metrics
    mean_edge = np.mean(edge_sums)
    metrics['edge_balance'] = np.max(edge_sums) / mean_edge if mean_edge > 0 else 0

    mean_in = np.mean(in_edge_sums)
    metrics['in_edge_balance'] = np.max(in_edge_sums) / mean_in if mean_in > 0 else 0

    mean_out = np.mean(out_edge_sums)
    metrics['out_edge_balance'] = np.max(out_edge_sums) / mean_out if mean_out > 0 else 0
    metrics['num_parts'] = num_parts
    metrics['num_edges'] = num_edges
    metrics['num_nodes'] = num_nodes
    metrics['local_edges'] = local_edges
    metrics['local_train_edges'] = local_train_edges
    metrics['remote_train_edges'] = remote_train_edges
    metrics['cut_edges'] = (num_edges - local_edges)
    metrics['edge_cut'] = (num_edges - local_edges) / num_edges
    metrics['train_edge_cut'] = (remote_train_edges / (remote_train_edges - local_train_edges)) if remote_train_edges - local_train_edges > 0 else 0
    metrics['vertex_balance'] = np.max(counts) / (num_nodes / num_parts)

    return metrics


def get_balance(v2p, num_parts):
    """
    Calculates the balance of partition sizes.

    Args:
        v2p (numpy.ndarray): Array mapping node IDs to partition IDs.
        num_parts (int): The total number of partitions.

    Returns:
        float: The balance of partition sizes.

    """
    unique_bins, counts = np.unique(v2p, return_counts=True)
    return np.max(counts) / (len(v2p) / num_parts)


def attach_features_to_graph(graph, features):
    """
    Attaches features to the nodes of a graph.

    Args:
        graph (Graph): The input graph.
        features (numpy.ndarray): Array of node features.

    Returns:
        Graph: The graph with features attached to the nodes.

    """
    graph.ndata['feat'] = torch.from_numpy(features)

    return graph