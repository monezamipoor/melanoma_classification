import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score

def print_raw_logits_and_probs(all_labels, all_outputs):
    # Compute probabilities
    logits = all_outputs
    probs = torch.sigmoid(logits)
    labels = all_labels

    # Convert to CPU + NumPy for easier handling
    logits_np = logits.squeeze().cpu().numpy()
    probs_np = probs.squeeze().cpu().numpy()
    labels_np = labels.squeeze().cpu().numpy()

    # Find indices of class 0 and class 1
    class0_indices = np.where(labels_np == 0)[0]
    class1_indices = np.where(labels_np == 1)[0]

    # Choose the smaller count to balance
    n = min(len(class0_indices), len(class1_indices), 10)  # limit to 10 max for readability

    # Randomly sample n indices from each class
    np.random.seed(42)  # for reproducibility
    sample0 = np.random.choice(class0_indices, n, replace=False)
    sample1 = np.random.choice(class1_indices, n, replace=False)

    # Combine and sort for cleaner output
    balanced_indices = np.concatenate([sample0, sample1])
    balanced_indices.sort()

    # Print results
    print("Balanced Samples:")
    for i in balanced_indices:
        print(f"Logit: {logits_np[i]:.4f}, Prob: {probs_np[i]:.4f}, Label: {labels_np[i]}")

def print_batch_label_dist(train_loader):
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # If labels are one-hot or float, convert to 0/1 int
        labels_int = labels.long() if labels.dtype != torch.long else labels
        class_counts = torch.bincount(labels_int, minlength=2)

        print(f"Batch {batch_idx + 1}: Class 0 = {class_counts[0].item()}, Class 1 = {class_counts[1].item()}")


def threshold_eval_metrics(probs, targets):
    thresholds = np.linspace(0.0, 1.0, 100)
    f1s, precisions, recalls = [], [], []

    for t in thresholds:
        preds = (probs > t).astype(int)
        f1s.append(f1_score(targets, preds))
        precisions.append(precision_score(targets, preds))
        recalls.append(recall_score(targets, preds))

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, f1s, label="F1 Score", color='blue')
    plt.plot(thresholds, precisions, label="Precision", color='green', linestyle='--')
    plt.plot(thresholds, recalls, label="Recall", color='red', linestyle='--')
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("F1 / Precision / Recall vs Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()