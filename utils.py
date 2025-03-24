from datetime import datetime
import os

def evaluate_metrics(opt, all_outputs, all_labels):
    # Skeleton function for evaluating metrics based on predictions and ground truths
    # Implement evaluation logic here using libraries like sklearn or torchmetrics
    # metrics is a dictionary like {'mAP' : 0.5, 'recall' : 0.9}
    metrics = {}
    return metrics

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