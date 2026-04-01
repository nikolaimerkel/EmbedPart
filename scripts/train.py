import argparse


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-graph_names', 
        nargs='+', 
        type=str, 
        help="The graph names to learn on",
        required=True
    )


    return parser
MODEL = "gat"
MODEL = "linkgraphsage"
MODEL = "graphsage"

LINKPREDICITION = False
if __name__ == "__main__":
    parser = parser()
    args = parser.parse_args()
    
    
    def template(graph_name, hardware, b, h, e, f):   
        l = len(f.split(" "))
        return f"python -m training.train --mode {hardware} --model_name {MODEL} --dataset {graph_name} --batch_size {b} --hidden_dims {h} --num_layers {l} --epochs {e} -fanout {f}"

    def template_linkprediction(graph_name, hardware, b, h, e, f,o):   
        l = len(f.split(" "))
        return f"python -m training.train_link --mode {hardware} --model_name {MODEL} --dataset {graph_name} --batch_size {b} --hidden_dims {h} --num_layers {l} --epochs {e} -fanout {f} --out_size {o}"

    commands = []
    
    #for h in [16,64,256]:
    for h in [64]:
        for f in ["25 10", "15 10 5"]:
            #for s in ["", ".rvs-0.1", ".rvs-0.3", ".rvs-0.5", ".rvs-0.7", ".rvs-0.9"]:
            for s in [""]:
                for graph_name in args.graph_names:
                    graph_name = graph_name + s
                    hardware = "puregpu"
                    if "papers" in graph_name:
                        hardware = "mixed"
                        
                    if LINKPREDICITION: 
                        for o in [8, 16]:
                            commands.append(template_linkprediction(
                            graph_name=graph_name, 
                            hardware=hardware,
                            b=8192, 
                            h=h,
                            e=101,
                            f=f,
                            o=o))
                    else:
                        commands.append(template(
                            graph_name=graph_name, 
                            hardware=hardware,
                            b=8192, 
                            h=h,
                            e=101,
                            f=f))
                
    with open("scripts/train.sh", "w") as f:
        f.write("\n".join(commands))
        