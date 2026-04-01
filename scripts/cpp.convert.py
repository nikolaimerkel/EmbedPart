import argparse


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-graph_names', 
        nargs='+', 
        type=str, 
        help="The graph names to convert",
        required=True
    )

    return parser


if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    
    def convert_tamplate(graph_name):
        return f"/mnt/data/lsh-partitioning/code/HGP/build/convert -filename /mnt/data/edgelists/{graph_name} -metis "
    
    commands = []
    for graph_name in args.graph_names:

        commands.append(convert_tamplate(graph_name))

    with open("scripts/cpp.convert.sh", "w") as f:
        f.write("\n".join(commands))
        