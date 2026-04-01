import argparse


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-graph_names', 
        nargs='+', 
        type=str, 
        help="The graph names to sparsifiy",
        required=True
    )


    return parser


if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    
    def template(graph_name, sparsifier, sparsifier_level):
        return f"python -m sparsification.sparsify -graph_name {graph_name} -sparsifier {sparsifier} -sparsifier_level {sparsifier_level}"

    commands = []
    for graph_name in args.graph_names:
        for sparsifier in [
            "rvs", 
           # "res"
            ]:
            for sparsifier_level in [0.1, 0.3, 0.5, 0.7, 0.9]:
                commands.append(template(graph_name, sparsifier, sparsifier_level))
    
    with open("scripts/sparsify.sh", "w") as f:
        f.write("\n".join(commands))
        