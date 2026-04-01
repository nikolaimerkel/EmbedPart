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
#data_papers100M = DglNodePropPredDataset(root="/mnt/data/.dgl", name='ogbn-papers100M')
data_papers100M = DglNodePropPredDataset(root="/mnt/data/test", name='ogbn-papers100M')
graph, labels = data_papers100M[0]

print(graph)

dataset = AsNodePredDataset(data_papers100M)
graph = dataset[0]
num_classes = data_papers100M.num_classes

print(graph)
        
        