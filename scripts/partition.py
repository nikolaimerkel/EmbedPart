import argparse
from datetime import datetime
import os


if __name__ == "__main__":
    commands = []
    
    GRAPHS = [
        "ogbn-arxiv", 
        "ogbn-products",
        "reddit",
        "ogbn-papers100M",
        "ogbl-citation2"
    ]
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # Create the directory name

    results_dir_base = f"/mnt/data/gnn-partitioning/results/partitioning-metrics/{timestamp}"

    # Create the directory
    os.makedirs(results_dir_base, exist_ok=True)

    def gnn_tamplate(graph_name_to_partition, graph_name_training, epochs, hidden, fanout, model_name):   
        l = len(fanout.split(" "))
        gnn_tamplate = [
            "python -m  partitioner.partition",
            "--mode mixed", 
            f"--model_name {model_name}", 
            "--batch_size 8192", 
            "--partitioner k-means", 
            "--num_parts 2 4 8 16 32", 
          #  "--num_parts 2 4", 
            f"--dataset_to_partition {graph_name_to_partition}", 
            f"--dataset {graph_name_training}", 
            f"--hidden_dims {hidden}", 
            f"--num_layers {l}", 
            f"--epochs {epochs}", 
            f"--fanout {fanout}", 
            f"--results_dir {results_dir_base}",
            f"--max_vertex_balance {1.1}",
            f"--max_training_balance {1.05}"
            ]
        return " ".join(gnn_tamplate)
    
    
    def gnn_tamplate_linkprediction(graph_name_to_partition, graph_name_training, epochs, hidden, fanout, model_name, out_size):   
        l = len(fanout.split(" "))
        gnn_tamplate = [
            "python -m  partitioner.partition",
           # "--mode mixed", #FOR papers100M
          #  "--mode puregpu", 
            "--mode cpu",
            f"--model_name {model_name}", 
            "--batch_size 8192", 
            "--partitioner k-means", 
            "--num_parts 2 4 8 16 32", 
          #  "--num_parts 2 4", 
            f"--dataset_to_partition {graph_name_to_partition}", 
            f"--dataset {graph_name_training}", 
            f"--hidden_dims {hidden}", 
            f"--num_layers {l}", 
            f"--epochs {epochs}", 
            f"--fanout {fanout}", 
            f"--results_dir {results_dir_base}",
            f"--max_vertex_balance {1.1}",
            f"--max_training_balance {1.05}",
            f"--out_size {out_size}"
            ]
        return " ".join(gnn_tamplate)
    
    
    
    if False:
        MODEL = "linkgraphsage"
        EPOCHS = [50,1,0] #3
        EPOCHS = [0,1,2,3,4,5,10,30,50,70,90] #11
        SPARSITY = [""]
        HIDDEN = [64] #1
        FANOUT = ["25 10", "15 10 5"]
        OUT_SIZE = [8,16]
        OUT_SIZE = [16]
    
        for e in EPOCHS:
            for h in HIDDEN:
                for f in FANOUT:
                    for s in SPARSITY:
                        for o in OUT_SIZE:
                            for graph_name in GRAPHS:
                                graph_name_to_partition = graph_name
                                graph_name_training = graph_name + s
                                commands.append(gnn_tamplate_linkprediction(
                                    graph_name_to_partition=graph_name_to_partition, 
                                    graph_name_training=graph_name_training,
                                    epochs=e, 
                                    hidden=h,
                                    fanout=f,
                                    model_name=MODEL,
                                    out_size=o))            
    
    if False:
        MODEL = "gat"
        MODEL = "graphsage"
        EPOCHS = [50] #11
        EPOCHS = [0,1,2,3,4,5,10,30,50,70,90] #11
        SPARSITY = [""]
        HIDDEN = [64] #1
        FANOUT = ["25 10", "15 10 5"]
    
        for e in EPOCHS:
            for h in HIDDEN:
                for f in FANOUT:
                    for s in SPARSITY:
                        for graph_name in GRAPHS:
                            if graph_name == "ogbl-citation2":
                                print(f"Skipping {graph_name} for now, as it is not supported by the nodepreidu model.")
                                continue  # Skip this graph for now
                            graph_name_to_partition = graph_name
                            graph_name_training = graph_name + s
                            commands.append(gnn_tamplate(
                                graph_name_to_partition=graph_name_to_partition, 
                                graph_name_training=graph_name_training,
                                epochs=e, 
                                hidden=h,
                                fanout=f,
                                model_name=MODEL))
    
    if False:
        
        MODEL = "graphsage"
        EPOCHS = [0,1,2,3,4,5,10,30,50,70,90] #11
        SPARSITY = [""]
        HIDDEN = [64] #1
        FANOUT = ["25 10", "15 10 5"] #2
        
        for e in EPOCHS:
            for h in HIDDEN:
                for f in FANOUT:
                    for s in SPARSITY:
                        for graph_name in GRAPHS:
                            graph_name_to_partition = graph_name
                            graph_name_training = graph_name + s
                            commands.append(gnn_tamplate(
                                graph_name_to_partition=graph_name_to_partition, 
                                graph_name_training=graph_name_training,
                                epochs=e, 
                                hidden=h,
                                fanout=f,
                                model_name=MODEL))
                            
        EPOCHS = [50] #1
        SPARSITY = [".rvs-0.1", ".rvs-0.3", ".rvs-0.5", ".rvs-0.7", ".rvs-0.9"] #5
        HIDDEN = [64] #1
        FANOUT = ["25 10", "15 10 5"] #2
        # = 11*1*2 + 5*2 = 22 + 10 = 32
        for e in EPOCHS:
            for h in HIDDEN:
                for f in FANOUT:
                    for s in SPARSITY:
                        for graph_name in GRAPHS:
                            graph_name_to_partition = graph_name
                            graph_name_training = graph_name + s
                            commands.append(gnn_tamplate(
                                graph_name_to_partition=graph_name_to_partition, 
                                graph_name_training=graph_name_training,
                                epochs=e, 
                                hidden=h,
                                fanout=f,
                                model_name=MODEL))
                                        
    # Run metis and random partitioner on all graphs
    for g in GRAPHS:
     #   commands.append(f"python -m  partitioner.partition --num_parts 2 4 8 16 32 --dataset_to_partition {g}  --partitioner random --results_dir {results_dir_base}")
        commands.append(f"python -m  partitioner.partition --num_parts 2 4 8 16 32 --dataset_to_partition {g}  --partitioner metis --results_dir {results_dir_base}")
      #  commands.append(f"python -m  partitioner.partition --num_parts 2 4 8 16 32 --dataset_to_partition {g}  --partitioner feature-partitioning --results_dir {results_dir_base}")

                        
                        
    print(f"Total number of Experiments: {len(commands)}")
                         
    with open("scripts/partition.sh", "w") as f:
        f.write("\n".join(commands))
        