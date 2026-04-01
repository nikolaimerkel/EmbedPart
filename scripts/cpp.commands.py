import argparse


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-graph_names', 
        nargs='+', 
        type=str, 
        help="The graph names to partition",
        required=True
    )

    return parser


if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    
    def ldg_template(graph_name, num_partitions):
        return f"/mnt/data/lsh-partitioning/code/HGP/build/gnn -filename /mnt/data/edgelists/{graph_name} -p {num_partitions} -partitioner ldg -write_out_vid2pid_file -directed"
    
    def spinner_template(graph_name, num_partitions):
        return f"/mnt/data/lsh-partitioning/code/HGP/build/gnn -filename /mnt/data/edgelists/{graph_name} -p {num_partitions} -partitioner spinner -write_out_vid2pid_file -directed"

    commands = []
    for graph_name in args.graph_names:
        for num_partitions in [2, 4, 8, 16, 32]:
            commands.append(ldg_template(graph_name, num_partitions))
            commands.append(spinner_template(graph_name, num_partitions))
            
    #for graph_name in args.graph_names:
     #   for num_partitions in [2, 4, 8, 16, 32]:
      #      commands.append(kahip_template(graph_name, num_partitions))
    
    with open("scripts/cpp.commands.sh", "w") as f:
        f.write("\n".join(commands))
        