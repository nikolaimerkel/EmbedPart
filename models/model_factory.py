from models.graphsage import GraphSage
from models.gat import GAT
from models.linkgraphsage import LinkGraphSage
def get_model(model_name, in_size, hid_size, out_size, n_layers):
    if model_name.lower() == "graphsage":
        return GraphSage(in_size, hid_size, out_size, n_layers)
    if model_name.lower() == "gat":
        return GAT(in_size, hid_size, out_size, n_layers)
    if model_name.lower() == "linkgraphsage":
        return LinkGraphSage(in_size, hid_size, out_size, n_layers)
    else:
        raise ValueError(f"Unknown model: {model_name}")