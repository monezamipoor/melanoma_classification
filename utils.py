import time
from datetime import datetime
import torch
import os
import pandas as pd
from numpy.f2py.auxfuncs import throw_error

EPOCH= 0
rundir = None



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

# YAML Nested Key Checker
def check_nested_key(data, keys):
    """Check if a nested key exists in a YAML dictionary."""
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return False
    return True

# create the run directory
def run_dir(opt=None):
    global rundir
    if rundir is not None:
        return rundir
    elif opt is None or check_nested_key(opt, ['testing', 'log_dir']) == False:
        return None
    else:
        os.makedirs(opt['testing']['log_dir'], exist_ok=True)
        path = os.path.join(opt['testing']['log_dir'], os.path.basename(opt['opt']).replace('.yml', '') + '-' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(path, exist_ok=True)
    rundir = path
    return rundir

def get_log_filename(opt):
    # If the log filename hasn't been set yet, compute and store it in opt
    if "log_filename" not in opt:
        log_dir = run_dir(opt)
        opt["log_filename"] = os.path.join(log_dir, "log.txt")
    return opt["log_filename"]

def log_model(opt, model):

    layers = []
    for name, param in model.named_parameters():
        layers.append({"Layer Name": name, "Shape": list(param.size()), "Parameters": param.numel()})

    log_dir = run_dir(opt)

    config_name = os.path.basename(opt['opt']).replace('.yml', '')
    fileout = os.path.join(log_dir, f"{opt['model']['backbone']}_{config_name}_{time.strftime('%Y%m%d-%H%M%S')}.csv")

    df = pd.DataFrame(layers)
    df.to_csv(fileout, index=False)

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

def save_checkpoint(opt, best_metrics, model, epoch, metrics):

    if check_nested_key(opt, ['testing', 'model_save_strategy']) == False or check_nested_key(opt, ['testing', 'model_save_metrics']) == False:
        return
    save_strategy = opt['testing']['model_save_strategy']
    save_metrics = opt['testing']['model_save_metrics']
    if save_strategy == 'none':
        return
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logdir = run_dir(opt)
    checkpoint_dir = opt['testing']['checkpoint_dir']
    os.makedirs(os.path.join(logdir, checkpoint_dir), exist_ok=True)

    if save_strategy == 'best':
        for metric in save_metrics:
            if metrics[metric] > best_metrics[metric]:
                best_metrics[metric] = metrics[metric]
                checkpoint = (timestamp +
                              '_epoch_' +
                              str(epoch) +
                              '_' +
                              os.path.basename(opt['opt']).replace('.yml', '') +
                              '_'+
                              metric +
                              '.pth')
                save_path = os.path.join(
                    logdir,
                    checkpoint_dir,
                    checkpoint
                )
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model for {metric} at epoch {epoch+1}")

    elif save_strategy == 'last':
        checkpoint = (timestamp +
                      '_epoch_' +
                      str(epoch) +
                      '_' +
                      os.path.basename(opt['opt']).replace('.yml', '') +
                      '_last' +
                      '.pth')
        save_path = os.path.join(
            logdir,
            checkpoint_dir,
            checkpoint
        )
        torch.save(model.state_dict(), save_path)
        print(f"Saved last model at epoch {epoch+1}")

    elif save_strategy == 'all':
        for metric in save_metrics:
            checkpoint = (timestamp +
                          '_epoch_' +
                          str(epoch) +
                          '_' +
                          os.path.basename(opt['opt']).replace('.yml', '') +
                          '_' +
                          metric +
                          '.pth')
            save_path = os.path.join(
                logdir,
                checkpoint_dir,
                checkpoint
            )
            torch.save(model.state_dict(), save_path)
            print(f"Saved model for {metric} at epoch {epoch+1}")