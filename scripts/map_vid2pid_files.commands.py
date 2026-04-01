

if __name__ == "__main__":

    
    commands = []
    
    # Be really careful with the paths here. 
    # .directed is used for ldg and spinner but not for kahip.
    
   # for graph_name in ["ogbn-arxiv", "ogbn-products", "reddit", "ogbn-papers100M", "ogbl-citation2"]:
    for graph_name in ["ogbl-citation2"]:
        for num_partitions in [2, 4 ,8, 16, 32]:
      #  for num_partitions in [8, 16, 32]:
      #  for num_partitions in [2,4]:
            for partitioner in ["ldg", "spinner"]:
            #for partitioner in ["spinner"]:
                cmd = " ".join([
                    "python /mnt/data/gnn-partitioning/scripts/map_vid2pid_files.py",
                    f"-bin2ascii /mnt/data/edgelists/{graph_name}.directed.bin2ascii",
                    f"-vid2pid /mnt/data/edgelists/{graph_name}.directed.{partitioner}.{num_partitions}.vid2pid",
                    f"-vid2pidOutput /mnt/data/partitioned/{graph_name}.{partitioner}.P{num_partitions}.vid2pid",
                ])
                commands.append(cmd)
            # Kahip is a bit different.
            # It uses the bidirected graph and the kahip partitioner.
            # ogbn-papers100M can not be partitioned with kahip.
         #   if graph_name not in ["ogbn-papers100M"]:
          #      cmd = " ".join([
           #         "python /mnt/data/gnn-partitioning/scripts/map_vid2pid_files.py",
            #        f"-bin2ascii /mnt/data/edgelists/{graph_name}.bidirected.bin2ascii",
             #       f"-vid2pid /mnt/data/edgelists/{graph_name}.bidirected.metis.kahipfsocial.{num_partitions}.vid2pid",
              #      f"-vid2pidOutput /mnt/data/partitioned/{graph_name}.kahipfsocial.P{num_partitions}.vid2pid",
               # ])
                #commands.append(cmd)
                
    
    with open("scripts/map_vid2pid_files.commands.sh", "w") as f:
        f.write("\n".join(commands))
        
        # ogbn-arxiv.directed.bin2ascii 
    # ogbn-arxiv.directed.ldg.2.vid2pid
    # ogbn-arxiv.directed.spinner.2.vid2pid
    
    # ogbn-papers100M.directed.bin2ascii
    # ogbn-papers100M.directed.ldg.2.vid2pid
    # ogbn-papers100M.directed.spinner.2.vid2pid
    
    # ogbn-products.directed.bin2ascii
    # ogbn-products.directed.ldg.2.vid2pid
    # ogbn-products.directed.spinner.2.vid2pid
    
    # reddit.directed.bin2ascii
    # reddit.directed.ldg.2.vid2pid
    # reddit.directed.spinner.2.vid2pid
    
    