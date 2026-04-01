import torch
import os
def save_model(model, optimizer, epoch, model_name, dataset_name, hid_size, n_layers, fanout,out_size=None):
    """
    Save the model checkpoint.
    """
    fanout = "-".join(map(str, fanout))
    print(fanout)
    if out_size is None:
        checkpoint_dir = f"checkpoints/{dataset_name}.{model_name}.H{hid_size}.L{n_layers}_F{fanout}/"
    else:
        checkpoint_dir = f"checkpoints/{dataset_name}.{model_name}.H{hid_size}.L{n_layers}_F{fanout}.O{out_size}/"
    os.makedirs(checkpoint_dir, exist_ok=True)  # Ensure directory exists
    
    checkpoint_path = os.path.join(checkpoint_dir, f"E{epoch}.pth")
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)

    print(f"Checkpoint saved: {checkpoint_path}")
    
def load_model(model, optimizer, epoch, model_name, dataset_name, hid_size, n_layers, fanout, out_size=None):
    fanout = "-".join(map(str, fanout))
    print(fanout)
    if out_size is None:
        checkpoint_dir = f"checkpoints/{dataset_name}.{model_name}.H{hid_size}.L{n_layers}_F{fanout}/"
    else:
        checkpoint_dir = f"checkpoints/{dataset_name}.{model_name}.H{hid_size}.L{n_layers}_F{fanout}.O{out_size}/"
    checkpoint_path = os.path.join(checkpoint_dir, f"E{epoch}.pth")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint for epoch {epoch} not found in {checkpoint_dir}.")

    print(f"Loading model from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return model, optimizer

    