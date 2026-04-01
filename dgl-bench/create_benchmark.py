

import os
from datetime import datetime


commands = []

EPOCHS_DIS_TRAINING = 20

fanout = {
    2: "25,10",
    3: "15,10,5"    
}
     

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# Create the directory name

results_dir_base = f"/mnt/data/gnn-partitioning/results/dist-gnn-training-metrics/{timestamp}"

# Create the directory
os.makedirs(results_dir_base, exist_ok=True)

        #  for model in ["SAGE"]:
        #  for model in ["SAGE"]:
        # for model in ["linkgraphsage"]:
#for model in ["SAGE", "GAT"]:
for p in [2,4]:
#for model in ["SAGE"]:
    for force_even in [False]:
        for num_layers in [2,3]:
       # for p in [4]:
           # for model in ["SAGE", "linkgraphsage"]:
           # for model in ["SAGE", "GAT"]:
            for model in ["SAGE"]:
          #  for num_layers in [3]:
            #for p in [2,4]:
                results_dir = f"{results_dir_base}/{model}"
                os.makedirs(results_dir, exist_ok=True)
                fan_out = fanout[num_layers]
                for graph in ["ogbn-arxiv", "ogbn-products", "reddit", "ogbn-papers100M"]:
                    for partitioner in [
                        "metis", 
                        "random", 
                        "ldg", 
                        "spinner",
                        f"D{graph}.E50.graphsage.H64.L2.F25-10.TB1.05.VB1.1", 
                        f"D{graph}.E50.graphsage.H64.L3.F15-10-5.TB1.05.VB1.1",
                        f"D{graph}.E50.linkgraphsage.H64.L2.O16.F25-10.TB1.05.VB1.1",
                        f"D{graph}.E50.linkgraphsage.H64.L3.O16.F15-10-5.TB1.05.VB1.1",
                        "directed.cuttana.cuttana256"
                        ]:
                    
                        train_script = "train_dist-link.py" if model in ["linkgraphsage"] else "train_dist.py"
                        BATCH_SIZE = 16*1024 if model in ["linkgraphsage"] else 1024 * 4
                        
                        fe = f"--force_even {force_even}" if force_even else ""
                        template = [
                            'python3 launch.py',
                            '--workspace /mnt/data/dgl-bench',
                            '--num_trainers 1',
                            '--num_samplers 0',
                            '--num_servers 1',
                            f'--part_config /mnt/data/partitioned/{graph}.{partitioner}.P{p}.vid2pid.d/{p}/{graph}.json',
                            f'--ip_config ip_config-{p}.txt',
                            '--ssh_username root',
                            '--ssh_port 2222',
                            f'"python3 {train_script} --graph_name {graph} --ip_config ip_config-{p}.txt --num_epochs {EPOCHS_DIS_TRAINING} --num_layers {num_layers} --fan_out {fan_out} --batch_size {BATCH_SIZE} --num_gpus 1 --results_dir {results_dir} --partitioner {partitioner} --num_parts {p} --model {model} {fe} "'
                        ]

                        commands.append(" ".join(template))


counter = 0
all = len(commands)
with open("create_benchmark.sh", "w") as f:
    for c in commands:
        f.write(f"echo {counter}/{all} \n")
        f.write(c + "\n")
        counter += 1

    
    
    
