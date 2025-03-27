from datetime import datetime
import torch
import os

EPOCH= 0


# def log_results(opt, metrics):
#     # Define log directory and filename
#     log_dir = opt['testing']['log_dir']
#     os.makedirs(log_dir, exist_ok=True)
#     log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{os.path.basename(opt['opt']).replace('.yml', '')}.txt")

#     # Write or append metrics to the log file
#     if not os.path.exists(log_filename):
#         with open(log_filename, 'w') as f:
#             # Write header
#             f.write("Epoch\t" + "\t".join(metrics.keys()) + "\n")

#     with open(log_filename, 'a') as f:
#         # Append metrics for each epoch
#         f.write(str(metrics.get('epoch', 'N/A')) + "\t" + "\t".join([str(v) for v in metrics.values()]) + "\n")
#     print(f"Metrics logged to {log_filename}")


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