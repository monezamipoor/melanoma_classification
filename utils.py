import time
from datetime import datetime
import torch
import os
import pandas as pd

EPOCH= 0

def get_log_filename(opt):
    # If the log filename hasn't been set yet, compute and store it in opt
    if "log_filename" not in opt:
        log_dir = opt['testing']['log_dir']
        os.makedirs(log_dir, exist_ok=True)
        # Create a run ID using the current time
        run_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # Optionally include the config name from opt if desired
        config_name = os.path.basename(opt['opt']).replace('.yml', '')
        opt["log_filename"] = os.path.join(log_dir, f"{run_id}_{config_name}.txt")
    return opt["log_filename"]

def log_results(opt, metrics):
    global EPOCH
    # If epoch is not provided, increment the global EPOCH counter and set it in metrics.
    if 'epoch' not in metrics or metrics.get('epoch') == 'N/A':
        EPOCH += 1
        metrics['epoch'] = EPOCH

    # Get (or compute) the log filename once
    log_filename = get_log_filename(opt)

    # If the log file doesn't exist yet, write a header
    if not os.path.exists(log_filename):
        with open(log_filename, 'w') as f:
            header = "Epoch\t" + "\t".join(metrics.keys()) + "\n"
            f.write(header)
    
    # Append metrics for the current epoch
    with open(log_filename, 'a') as f:
        f.write(str(metrics.get('epoch')) + "\t" +
                "\t".join([str(v) for v in metrics.values()]) + "\n")
    
    print(f"Metrics logged to {log_filename}")



# Determine cuda and use this as a way to configure any device params
# opt is passed but not currently used
def cuda_available(opt):
    # GPU operations have a separate seed we also want to set
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Additionally, some operations on a GPU are implemented stochastic for efficiency
    # We want to ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device", device)

    return device

def log_model(opt, model):

    layers = []
    for name, param in model.named_parameters():
        layers.append({"Layer Name": name, "Shape": list(param.size()), "Parameters": param.numel()})

    log_dir = opt['testing']['log_dir']
    os.makedirs(log_dir, exist_ok=True)

    config_name = os.path.basename(opt['opt']).replace('.yml', '')
    fileout = os.path.join(log_dir, f"{opt['model']['backbone']}_{config_name}_{time.strftime("%Y%m%d-%H%M%S")}.csv")

    df = pd.DataFrame(layers)
    df.to_csv(fileout, index=False)