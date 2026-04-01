import dgl
import torch
import random
import os



class GraphSparsifier:
    """
    A class to perform graph sparsification using various methods, including random vertex and edge removal and degree-based node removal.
    """

    def __init__(self, graph, base_path, graph_name):
        self.graph = graph
        self.sparsified_graph = graph.clone()
        self.base_path = f"{base_path}/{graph_name}"
        
        # check if the directory exists
     #   if not os.path.exists(self.base_path):
      #      os.makedirs(self.base_path)
             
    def is_cached(self, path):
        return os.path.exists(path)
    
    def load_cached_graph(self, path):
        self.sparsified_graph = dgl.load_graphs(path)[0][0]
        print("Loaded graph from cache:\n", self.sparsified_graph )
    
    def cache_graph(self, filename):
        dgl.save_graphs(filename, [self.sparsified_graph])   
        print("Cached graph:\n", self.sparsified_graph)
        
    def random_vertex_sparsifier(self, keep_prob, cache=True):
        """
        Sparsifies the graph by randomly removing vertices based on the specified probability.

        Parameters:
        - keep_prob (float): The probability of keeping each node in the graph (0 < keep_prob <= 1).

        Returns:
        - sparsified_graph (dgl.DGLGraph): A sparsified version of the original graph with fewer nodes.
        """
        file_path = f"{self.base_path}.rvs-{keep_prob}.dgl" 
        if self.is_cached(path=file_path):
            self.load_cached_graph(path=file_path)
        else:
            self.sparsified_graph = self.graph.clone()
            number_of_nodes_to_remove = int((1.0 - keep_prob) * self.sparsified_graph.number_of_nodes())
            perm = torch.randperm(self.sparsified_graph.number_of_nodes())
            nodes_to_remove = perm[:number_of_nodes_to_remove]
            self.sparsified_graph.remove_nodes(nodes_to_remove)
            if cache:
                self.cache_graph(filename=file_path)
        return self.sparsified_graph

       
    def random_edge_sparsifier(self, keep_prob, cache=True):
        """
        Sparsifies the graph by randomly removing edges based on the specified probability.

        Parameters:
        - graph (dgl.DGLGraph): The input graph to be sparsified.
        - keep_prob (float): The probability of keeping each edge in the graph (0 < keep_prob <= 1).

        Returns:
        - sparsified_graph (dgl.DGLGraph): A sparsified version of the original graph with fewer edges.
        """
        file_path = f"{self.base_path}.res-{keep_prob}.dgl" 
        if self.is_cached(path=file_path):
            self.load_cached_graph(path=file_path)
        else: 
            self.sparsified_graph = self.graph.clone()
            number_of_edges_to_remove = int((1.0 - keep_prob) * self.sparsified_graph.number_of_edges())
            perm = torch.randperm(self.sparsified_graph.number_of_edges())
            edges_to_remove = perm[:number_of_edges_to_remove]
            self.sparsified_graph.remove_edges(edges_to_remove)
            if cache:
                self.cache_graph(filename=file_path)      
        return self.sparsified_graph
   
    def degree_based_sparsifier(self, degree_threshold, cache=True):
        """
        Sparsifies the graph by removing nodes with a degree below the specified threshold.

        Parameters:
        - graph (dgl.DGLGraph): The input graph to be sparsified.
        - degree_threshold (int): The degree threshold below which nodes are removed.

        Returns:
        - sparsified_graph (dgl.DGLGraph): A sparsified version of the original graph with nodes below the degree threshold removed.
        """
        file_path = f"{self.base_path}.dbs-{degree_threshold}.dgl" 
        if self.is_cached(path=file_path):
            self.load_cached_graph(path=file_path)
        else:
            self.sparsified_graph = self.graph.clone()
            degrees = self.sparsified_graph.in_degrees() + self.sparsified_graph.out_degrees()  # Total degree
            print(degrees.max(), degrees.min())
            nodes_to_remove = torch.nonzero(degrees < degree_threshold, as_tuple=False).squeeze()
            self.sparsified_graph.remove_nodes(nodes_to_remove)
            if cache:
                self.cache_graph(filename=file_path)
        return self.sparsified_graph
    
    def gap_sparsifier(self, s, cache=True):
        file_path = f"{self.base_path}.gap-{s}.dgl" 
        if self.is_cached(path=file_path):
            self.load_cached_graph(path=file_path)
        else: 
            self.sparsified_graph = self.graph.clone()              
            mean_degree = 2 * self.sparsified_graph.number_of_edges() / self.sparsified_graph.number_of_nodes()
            in_degrees = self.sparsified_graph.in_degrees()
            out_degrees = self.sparsified_graph.out_degrees()
            eids_to_delete = []
            for dst in self.sparsified_graph.dstnodes():
                sources = self.sparsified_graph.predecessors(dst.item())
                for src in sources:
                    random_number = random.random()
                    remove_prob = (s * mean_degree) / min(out_degrees[src.item()].item(), in_degrees[dst.item()].item())
                    if random_number > remove_prob:
                        eids_to_delete.append(self.sparsified_graph.edge_ids(src, dst))
            self.sparsified_graph.remove_edges(eids_to_delete)
            print(f"Deleted {len(eids_to_delete)} of the {self.sparsified_graph.number_of_edges()} egdes")
            if cache:
                self.cache_graph(filename=file_path)
            
        
        return self.sparsified_graph
            
       
       
if __name__ == "__main__":
    graph = dgl.rand_graph(1000, 10000)
    gs = GraphSparsifier(graph=graph, graph_name="ER")
    print(graph)
    print("RVS", gs.random_vertex_sparsifier(keep_prob=0.7))
    print("RES", gs.random_edge_sparsifier(keep_prob=0.7))
    print("DEG BASED",gs.degree_based_sparsifier(degree_threshold=10))
    print("GAP-0.1", gs.gap_sparsifier(s=0.1))
    #print("GAP-0.3", gs.gap_sparsifier(graph, 0.3))
    #print("GAP-0.5", gs.gap_sparsifier(graph, 0.5))
    #print("GAP-0.7",gs.gap_sparsifier(graph, 0.7))
    #print("GAP-0.9",gs.gap_sparsifier(graph, 0.9))
    print(graph)