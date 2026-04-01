import dgl
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data import RedditDataset
from dgl.data import AsNodePredDataset
from dgl.data import CoraGraphDataset
from dgl.data import CiteseerGraphDataset
from dgl.data import PubmedGraphDataset
from dgl.data import AmazonCoBuyComputerDataset
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from dgl.data import YelpDataset
import torch as th

from configs.config import DGL_RAW_GRAPHS_DOWNLOAD, DGL_GRAPHS

#def get_sparsifier_and_level(graph_name, base_name):  
#    # reddit.rvs-0.1
#    print("graph_name", graph_name)
#    
#    s_l = graph_name.split(base_name + ".")[-1]
#
#       
#    sparsifier = s_l.split("-")[-2]
#    if sparsifier == "res":
#        sparsifier = "edge-sparsification"
#    if sparsifier == "rvs":
#        sparsifier = "vertex-sparsification"
#    sparsifier_level = s_l.split("-")[-1]
#    return sparsifier, sparsifier_level  

def get_graph_and_dataset(graph_name, base_name):
   # sparsifier, sparsifier_level = get_sparsifier_and_level(graph_name=graph_name, base_name=base_name)
    print("about to load sparsified graph", graph_name, base_name)
    path = f"{DGL_GRAPHS}/{graph_name}.dgl"
    print("we are about to load:", path)
    graph = dgl.load_graphs(path)[0][0]     
    return graph  

def load_dgl_graph(graph_name):
    print(f"About to load graph: {graph_name}")
   # graph_name = graph_name.lower()
    
    base_name = "reddit".lower()
    if base_name in graph_name.lower():
        data_reddit = RedditDataset(raw_dir=DGL_RAW_GRAPHS_DOWNLOAD+"/reddit")
        dataset = AsNodePredDataset(data_reddit)
        graph = dataset[0]
        num_classes = data_reddit.num_classes
        
        if not base_name == graph_name.lower():
            graph = get_graph_and_dataset(graph_name, base_name)    
            
        return graph, num_classes
    
    base_name = 'ogbn-products'.lower()
    if base_name in graph_name.lower(): 
        data_products = DglNodePropPredDataset(root=DGL_RAW_GRAPHS_DOWNLOAD, name='ogbn-products')
        dataset = AsNodePredDataset(data_products)
        graph = dataset[0]
        num_classes = data_products.num_classes
        
        if not base_name == graph_name.lower():
            graph = get_graph_and_dataset(graph_name, base_name)    
            
        return graph, num_classes
    
    base_name = 'ogbl-citation2'.lower()
    if base_name in graph_name.lower(): 
        dataset_citation2 = DglLinkPropPredDataset(root=DGL_RAW_GRAPHS_DOWNLOAD, name="ogbl-citation2")
        graph = dataset_citation2[0]
        num_classes = 1 # we do not have node labels for this graph, only edge labels

        if not base_name == graph_name.lower():
            graph = get_graph_and_dataset(graph_name, base_name)    
            
        return graph, num_classes
    
    base_name = 'ogbn-arxiv'.lower()
    if base_name in graph_name.lower(): 
        data_products = DglNodePropPredDataset(root=DGL_RAW_GRAPHS_DOWNLOAD,name='ogbn-arxiv')
        dataset = AsNodePredDataset(data_products)
        graph = dataset[0]
        num_classes = data_products.num_classes
        
        if not base_name == graph_name.lower():
            graph = get_graph_and_dataset(graph_name, base_name)    
            
        return graph, num_classes        
        
    base_name = 'ogbn-papers100M'.lower()
    if base_name in graph_name.lower():
        print("Start loading", graph_name)
        data_papers100M = DglNodePropPredDataset(root=DGL_RAW_GRAPHS_DOWNLOAD, name='ogbn-papers100M') 
        print("Stop loading")
        graph, labels = data_papers100M[0]
        graph.ndata['label'] = labels

        print("graph", graph)
        # Add train/val/test masks
        split_idx = data_papers100M.get_idx_split()
        train_idx = split_idx['train']
        val_idx = split_idx['valid']
        test_idx = split_idx['test']

        # Initialize boolean masks
        graph.ndata['train_mask'] = th.zeros(graph.num_nodes(), dtype=th.bool)
        graph.ndata['val_mask'] = th.zeros(graph.num_nodes(), dtype=th.bool)
        graph.ndata['test_mask'] = th.zeros(graph.num_nodes(), dtype=th.bool)

        # Assign split indices to masks
        graph.ndata['train_mask'][train_idx] = True
        graph.ndata['val_mask'][val_idx] = True
        graph.ndata['test_mask'][test_idx] = True

        num_classes = data_papers100M.num_classes

        if not base_name == graph_name.lower():
            graph = get_graph_and_dataset(graph_name, base_name)   
            print("We used sparsification and the graph has this size", graph)
        
        graph.ndata['label'] = graph.ndata['label'].squeeze(1)
        
        
        
        return graph, num_classes
    
    else:
        raise ValueError("Unknown graph name")
    
    
    