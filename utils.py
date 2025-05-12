"""
Define a set of utility functions for the training and evaluation of models.
All functions are called when the other modules call them.

"""


import shutil
import time
from datetime import datetime
import numpy as np
import torch
import os
import pandas as pd
from matplotlib import pyplot as plt
from numpy.f2py.auxfuncs import throw_error
import math
import torch.nn.functional as F

EPOCH= 0
rundir = None

# Determine cuda and use this as a way to configure any device params
# opt is passed but not currently used
def cuda_available(opt):
    """Check if CUDA is available and set the device accordingly."""
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
        # Check if the current key exists in the data
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return False
    return True

# create the run directory
def run_dir(opt=None):
    """
    Create a run directory for saving logs and checkpoints. 
    
    """
    global rundir
    if rundir is not None:
        return rundir
    elif opt is None or check_nested_key(opt, ['testing', 'log_dir']) == False:
        return None
    else:
        os.makedirs(opt['testing']['log_dir'], exist_ok=True)
        path = os.path.join(opt['testing']['log_dir'], os.path.basename(opt['opt']).replace('.yml', '') + '-' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        os.makedirs(path, exist_ok=True)

        # Copy the YML over for posterity
        if os.path.exists(opt['opt']):
            if not os.path.exists(os.path.join(path, os.path.basename(opt['opt']))):
                shutil.copy(opt['opt'], os.path.join(path, os.path.basename(opt['opt'])))
        else:
            print("Unable to copy config file to log output")

    rundir = path
    return rundir

def get_log_filename(opt):
    """
    Get the log filename for the current run. If it doesn't exist, create it.
    
    """
    # If the log filename hasn't been set yet, compute and store it in opt
    if "log_filename" not in opt:
        log_dir = run_dir(opt)
        opt["log_filename"] = os.path.join(log_dir, "log.txt")
    return opt["log_filename"]

def log_model(opt, model):
    """
    Log the model architecture and parameters to a CSV file.
    
    """

    layers = []
    for name, param in model.named_parameters():
        layers.append({"Layer Name": name, "Shape": list(param.size()), "Parameters": param.numel()})

    log_dir = run_dir(opt)

    config_name = os.path.basename(opt['opt']).replace('.yml', '')
    fileout = os.path.join(log_dir, f"{opt['model']['backbone']}_{config_name}_{time.strftime('%Y%m%d-%H%M%S')}.csv")

    df = pd.DataFrame(layers)
    df.to_csv(fileout, index=False)

def log_results(opt, metrics, phase='val', tag='notag'):
    """
    Log the results of the current epoch to a CSV file.
    
    """
    # Get (or compute) the log filename once
    log_dir = run_dir(opt)
    log_filename = os.path.join(log_dir, f"log_{phase}_{tag}.csv")   # <== DIFFERENT FILE PER TAG

    # If the log file doesn't exist yet, write a header
    if not os.path.exists(log_filename):
        with open(log_filename, 'w') as f:
            header = ",".join(metrics.keys()) + "\n"
            f.write(header)

    # Append metrics for the current epoch
    with open(log_filename, 'a') as f:
        f.write(",".join([str(v) for v in metrics.values()]) + "\n")

def log_test(opt, metrics, tag='notag'):
    log_results(opt, metrics, phase='test', tag=tag)


def save_checkpoint(opt, best_metrics, model, epoch, metrics, fold=None):
    """
    Save the model checkpoint based on the specified strategy.
    
    """
    # utils.py
    if not metrics:
        # If metrics is empty, we are in contrastive phase -> SKIP checkpoint saving
        print("Skipping checkpoint saving (no metrics yet in contrastive phase).")
        return None


    if check_nested_key(opt, ['testing', 'model_save_strategy']) == False or check_nested_key(opt, ['testing', 'model_save_metrics']) == False:
        return None
    save_strategy = opt['testing']['model_save_strategy']
    save_metrics = opt['testing']['model_save_metrics']
    if save_strategy == 'none':
        return None
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logdir = run_dir(opt)
    checkpoint_dir = opt['testing']['checkpoint_dir']
    os.makedirs(os.path.join(logdir, checkpoint_dir), exist_ok=True)

    save_path = None

    if fold is None:
        fold_and_epoch = '-NF' + '_E' + str(epoch) + '-'
    else:
        fold_and_epoch = '-F' + str(fold) + '_E' + str(epoch) + '-'

    if save_strategy == 'best':
        for metric in save_metrics:
            if metrics[metric] > best_metrics[metric]:
                best_metrics[metric] = metrics[metric]
                checkpoint = (timestamp +
                              fold_and_epoch +
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
                print(f"Saved best model for {metric} at epoch {epoch}")

    elif save_strategy == 'last':
        checkpoint = (timestamp +
                      fold_and_epoch +
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
                          fold_and_epoch +
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
            print(f"Saved model for {metric} at epoch {epoch}")

    return save_path


def get_checkpoint_dir(opt):
    """
    Get the checkpoint directory for saving model checkpoints.
    
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    logdir = run_dir(opt)
    checkpoint_dir = opt['testing']['checkpoint_dir']
    os.makedirs(os.path.join(logdir, checkpoint_dir), exist_ok=True)

    return checkpoint_dir, logdir, timestamp


def check_dataset_balance(dataset, label_name='Label'):
    """
    Check the distribution of labels in the dataset and print the counts.
    
    """
    from collections import Counter
    labels = [int(label) for _, label in dataset]
    count = Counter(labels)
    total = sum(count.values())

    print(f"📊 {label_name} distribution:")
    for cls, num in count.items():
        pct = 100 * num / total
        print(f"  Class {cls}: {num} samples ({pct:.2f}%)")


def check_prediction_distribution(model, dataloader, device='cuda'):
    """
    Check the distribution of predictions from the model on a given dataloader.
    
    """
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).float()

            logits = model(images)
            if logits.shape[-1] != 1:
                logits = logits.squeeze(-1)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).int()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.int().cpu().numpy())

            # Stop after one batch
            break

    from collections import Counter
    pred_counts = Counter(all_preds)
    label_counts = Counter(all_targets)

    print("🔍 First batch target distribution:", dict(label_counts))
    print("🔍 First batch predicted distribution:", dict(pred_counts))

# Takes probabilities between 0 and 1
def write_kaggle_csv(opt, images, probs, tag="natural"):
    log_dir = run_dir(opt)

    image_list = [item.replace('.jpg', '') for item in images.tolist()]

    df = pd.DataFrame({
        'image_name': image_list,
        'target': probs.tolist()
    })

    config_name = os.path.basename(opt['opt']).replace('.yml', '')
    fileout = os.path.join(log_dir, f"{opt['model']['backbone']}_{config_name}_predictions_{tag}.csv")

    df.to_csv(fileout, index=False)

def soft_voting_probs_from_logits(ensemble_logits):
    probs = torch.sigmoid(ensemble_logits)
    avg_probs = probs.mean(dim=0)
    return avg_probs

def save_augmented_samples(loader, num_samples=10, save_dir="/content/drive/MyDrive/melanoma_classification/logs/Sample"):

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get one batch from the DataLoader. Assuming batch[0] contains the images.
    batch = next(iter(loader))
    images = batch[0]  # (B, C, H, W)

    # Define the normalization parameters used in your transforms:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    imgs_denorm = []
    for i in range(num_samples):
        img = images[i].clone().cpu()
        img = denormalize_image(img, imagenet_mean, imagenet_std)
        # Convert from (C, H, W) to (H, W, C)
        img_np = img.permute(1, 2, 0).numpy()
        # Clip values to [0, 1] for display purposes
        img_np = np.clip(img_np, 0, 1)
        imgs_denorm.append(img_np)

    # Create a grid plot for the samples
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    for idx, ax in enumerate(axes):
        ax.imshow(imgs_denorm[idx])
        ax.set_title(f"Sample {idx+1}")
        ax.axis("off")
    plt.suptitle("Augmented Samples")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = os.path.join(save_dir, f"augmented_samples_{timestamp}.png")

    # Save the figure to the unique file path
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved augmented samples to {save_path}")


def denormalize_image(tensor, mean, std):

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def denormalize_image(tensor, mean, std):
    """Undo ImageNet normalization on a single C×H×W tensor."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def save_augmented_samples(loader, num_samples=10, save_dir=None, ncols=10):
    # 1) collect exactly num_samples images
    imgs = []
    for batch in loader:
        batch_imgs = batch[0]    # assumes loader yields (images, labels)
        for img in batch_imgs:
            imgs.append(img.clone().cpu())
            if len(imgs) >= num_samples:
                break
        if len(imgs) >= num_samples:
            break

    if not imgs:
        print("  No images found in loader!")
        return
    imgs = imgs[:num_samples]

    # 2) denormalize each
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    imgs_denorm = []
    for img in imgs:
        img = denormalize_image(img, imagenet_mean, imagenet_std)
        img_np = img.permute(1, 2, 0).numpy()
        imgs_denorm.append(np.clip(img_np, 0, 1))

    # 3) set up grid
    nrows = math.ceil(num_samples / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < len(imgs_denorm):
            ax.imshow(imgs_denorm[idx])
        ax.axis("off")
    plt.suptitle("Augmented Samples", y=1.02)

    # 4) save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"augmented_samples_{timestamp}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved augmented samples to {save_path}")
    else:
        plt.show()
        plt.close(fig)


    #print(f"Kaggle CSV saved to {fileout}")

# Establish the mean of a tensor of logits. Principally for soft-voting.
# Input must be [x,y] x = row per model, y = logits to average. E.g. [2,500] would be 2 models with 500 logits each.
# Note that this also works for single model (x=1) inputs as (y / 1 = y)
def soft_voting_probs_from_logits(ensemble_logits):
    probs = torch.sigmoid(ensemble_logits)
    avg_probs = probs.mean(dim=0)

    #print(f"[DEBUG] voting input shape: {probs.shape}, mean dim=0 -> divides by: {probs.shape[0]}")
    #print(f"[DEBUG] voting output shape: {avg_probs.shape}")

    return avg_probs

def save_augmented_samples(loader, num_samples=10, save_dir="/content/drive/MyDrive/melanoma_classification/logs/Sample"):

    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get one batch from the DataLoader. Assuming batch[0] contains the images.
    batch = next(iter(loader))
    images = batch[0]  # (B, C, H, W)

    # Define the normalization parameters used in your transforms:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    imgs_denorm = []
    for i in range(num_samples):
        img = images[i].clone().cpu()
        img = denormalize_image(img, imagenet_mean, imagenet_std)
        # Convert from (C, H, W) to (H, W, C)
        img_np = img.permute(1, 2, 0).numpy()
        # Clip values to [0, 1] for display purposes
        img_np = np.clip(img_np, 0, 1)
        imgs_denorm.append(img_np)

    # Create a grid plot for the samples
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    for idx, ax in enumerate(axes):
        ax.imshow(imgs_denorm[idx])
        ax.set_title(f"Sample {idx+1}")
        ax.axis("off")
    plt.suptitle("Augmented Samples")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = os.path.join(save_dir, f"augmented_samples_{timestamp}.png")

    # Save the figure to the unique file path
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved augmented samples to {save_path}")


def denormalize_image(tensor, mean, std):

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def build_metadata(df, sex_map, site_map):
    """
    Given a DataFrame with columns ['sex','age_approx','anatom_site_general_challenge'],
    plus precomputed maps sex_map and site_map,
    returns a list of [sex,age,site_onehot…] FloatTensors.
    """
    num_sites = 6
    metas = []
    for _, row in df.iterrows():
        s = sex_map[row.sex]
        a = float(row.age_approx)
        idx = site_map[row.anatom_site_general_challenge]
        onehot = F.one_hot(torch.tensor(idx, dtype=torch.long),
                            num_classes=num_sites).float()
        vec = torch.cat([
            torch.tensor([s, a], dtype=torch.float32),
            onehot
        ], dim=0)
        metas.append(vec)
    return metas
