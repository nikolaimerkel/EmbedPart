
if __name__ == "__main__":

    commands = []
    
    
    partitions =  [2,4,8,16,32]
    for p in partitions:
        #for graph in ["ogbn-arxiv", "ogbn-products", "reddit", "ogbn-papers100M", "ogbl-citation2"]:
        for graph in ["ogbn-arxiv", "ogbn-products", "reddit", "ogbn-papers100M"]:
       # for graph in ["ogbl-citation2"]:
            for partitioner in ["directed.cuttana.cuttana256"]:
                # python -m scripts.compute_partition_metrics_extern_no_cpp 
                # -graph /mnt/data/dgl/reddit.dgl 
                # -vid2pid /mnt/data/edgelists/reddit.directed.cuttana.cuttana256.P4  
                # -num_parts 4 -
                # partitioner cuttana 
                # -graph_name reddit

                t = [
                "python -m scripts.compute_partition_metrics_extern_no_cpp",
                    f"-graph /mnt/data/dgl/{graph}.dgl",
                    f"-vid2pid /mnt/data/partitioned/{graph}.{partitioner}.P{p}.vid2pid",
                    f"-num_parts {p}",
                    f"-partitioner {partitioner}",
                    f"-graph_name {graph}"
                ]
                commands.append(" ".join(t))
                
                # -graph /mnt/data/dgl/ogbn-arxiv.dgl 
                # -vid2pid /mnt/data/partitioned/ogbn-arxiv.Dogbn-arxiv.E50.graphsage.H64.L3.F15-10-5.P2.vid2pid 
                # -cpp_metrics /mnt/data/edgelists/ogbn-arxiv.directed.ldg.2.edgecut.partitioning.metrics.json 
                # -num_parts 2

    with open("scripts/compute_partition_metrics_extern_no_cpp.sh", "w") as f:
        f.write("\n".join(commands))
        