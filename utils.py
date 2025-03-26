from datetime import datetime
import torch
import os


def log_results(opt, metrics):
    # Define log directory and filename
    log_dir = opt['testing']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{os.path.basename(opt['opt']).replace('.yml', '')}.txt")

    # Write or append metrics to the log file
    if not os.path.exists(log_filename):
        with open(log_filename, 'w') as f:
            # Write header
            f.write("Epoch\t" + "\t".join(metrics.keys()) + "\n")

    with open(log_filename, 'a') as f:
        # Append metrics for each epoch
        f.write(str(metrics.get('epoch', 'N/A')) + "\t" + "\t".join([str(v) for v in metrics.values()]) + "\n")
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