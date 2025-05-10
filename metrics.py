'''
metrics.py - Calculates evaluation metrics based on targets and either logits, probabilities or predictions
'''

import os
import time

import torch
from sentry_sdk.utils import epoch

from sklearn.metrics import confusion_matrix,  roc_curve, auc, average_precision_score, roc_auc_score, precision_recall_curve

import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_accuracy,
    binary_f1_score,
    binary_precision,
    binary_recall,
    binary_average_precision,
    binary_confusion_matrix
)

import utils
from debug import threshold_eval_metrics
from wandb_helper import wandb_log_cm


# For automatic detection of probability thresholds when configured
def find_best_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = f1.argmax()
    return thresholds[best_idx], precision[best_idx], recall[best_idx], f1[best_idx]


# WARNING: LOGITS, PROBABILITIES AND PREDICTIONS ARE NOT THE SAME THING.
# LOGITS: Raw loss
# PROBS: Sigmoid activated probability from loss, resulting in float between 0 and 1.
# PREDS: A hard 0 or 1, effectively thresholded PROBS.
#
# evaluate_metrics now expects probs, not logits. BUT WHY?
# ... torchmetrics can use both logits and probs
# ... ensemble voting of multiple models needs to be done on probabilities (or preds) as so the input from these CANT be logits
# ... so we go with the lowest common denominator for all of our test choices. Which is probabilities.
def evaluate_metrics(opt, probs, target, epoch, tag=None):

    #debug - Uncomment to show spread of probs in debug
    #threshold_eval_metrics(probs, target)
    #end debug

    valdict = {'auc':'AUC','accuracy':'Accuracy', 'precision':'Precision', 'recall':'Recall', 'f1':'F1 Score', 'ap':'Average Precision', 'map':'mAP'}
    testdict = {'auc':'T_AUC','accuracy':'T_Accuracy', 'precision':'T_Precision', 'recall':'T_Recall', 'f1':'T_F1 Score', 'ap':'T_Average Precision', 'map':'T_mAP'}

    # Helps separate results in wandb and log to a separate test log file
    if epoch == 'Test':
        rundict = testdict.copy()
    else:
        rundict = valdict.copy()

    # Check we likely have probabilities rather than logits. Safety check.
    if not ((probs >= 0 - 1e-6).all() and (probs <= 1 + 1e-6).all()):
        print("-------")
        print("WARNING: Evaluation Metrics may have been passed logits rather than probabilities.")
        print("min:", probs.min().item())
        print("max:", probs.max().item())
        print("mean:", probs.mean().item())
        print("std:", probs.std().item())
        print("-------")

    results = {}
    classlabels = ["Benign", "Malignant"]
    threshold_value = opt['testing']['threshold_value']

    # Automatic calculation of thresholds if set in config.
    if threshold_value == 'auto':
        # Find best threshold based on F1 score
        threshold_value, best_precision, best_recall, best_f1 = find_best_threshold(target.cpu().numpy(), probs.cpu().numpy())
        threshold_value = float(threshold_value)
        print(f"Auto-selected best threshold: {threshold_value:.4f} (Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f})")
    else:
        print(f"Using configured threshold: {threshold_value}")

    # Convert probabilities to binary predictions using a given threshold.
    print(f"Threshold equals to: {threshold_value}")
    preds_binary = (probs > threshold_value).int()
    target_int = target.int()

    # Retrieve the list of metrics to compute from your configuration.
    config_metrics = opt['testing']['model_metrics']
    def to_scalar(x):
        return x.item() if isinstance(x, torch.Tensor) else x

    # Add the supplied epoch to the metrics collection
    results['epoch'] = epoch

    # Loop over each metric name and compute the corresponding metric.
    for metric in config_metrics:
        metric_lower = metric.lower()
        if metric_lower == 'auc':
            results[rundict.get(metric_lower, metric_lower)] = round(to_scalar(binary_auroc(probs, target, thresholds=None)), 4)
        elif metric_lower == 'accuracy':
            results[rundict.get(metric_lower, metric_lower)] = round(to_scalar(binary_accuracy(probs, target, threshold=threshold_value)), 4)
        elif metric_lower == 'precision':
            results[rundict.get(metric_lower, metric_lower)] = round(to_scalar(binary_precision(probs, target, threshold=threshold_value)), 4)
        elif metric_lower == 'recall':
            results[rundict.get(metric_lower, metric_lower)] = round(to_scalar(binary_recall(probs, target, threshold=threshold_value)), 4)
        elif metric_lower == 'f1':
            results[rundict.get(metric_lower, metric_lower)] = round(to_scalar(binary_f1_score(probs, target, threshold=threshold_value)), 4)
        elif metric_lower == 'ap':
            results[rundict.get(metric_lower, metric_lower)] = round(to_scalar(binary_average_precision(probs, target)), 4)
        elif metric_lower == 'map':
            results[rundict.get(metric_lower, metric_lower)] = average_precision_score(target_int.numpy(), probs.numpy())
        elif metric_lower =='cm':
            cm = confusion_matrix(target_int.numpy(), preds_binary.numpy())
            results[rundict.get(metric_lower, metric_lower)] = cm

            wandb_log_cm(preds_binary.cpu().numpy().flatten().tolist(), target_int.cpu().numpy().flatten().tolist(), classlabels, "Confusion Matrix - Epoch: " + str(epoch))

    # Log the confusion matrix and ROCAUC for the Test epoch only
    if tag is not None:
        # Plot and log the Confusion Matrix
        visualize_confusion_matrix(confusion_matrix(target_int.numpy(), preds_binary.numpy()), classlabels, "Confusion Matrix - Test: " + tag, tag)

        # And the ROC
        fpr, tpr, roc_thresholds = roc_curve(target_int.numpy(), probs.numpy())
        roc_auc_val = auc(fpr, tpr)
        visualize_roc_curve(fpr, tpr, roc_auc_val, tag)

    return results

# Writes a plot of a CM to the log directory
def visualize_confusion_matrix(cm, labels=['Negative', 'Positive'], title='Confusion Matrix', tag=''):
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.tight_layout()


    rundir = utils.run_dir()

    filepth = os.path.join(rundir, f"confusion_matrix_{tag}_{time.strftime('%Y%m%d-%H%M%S')}.png")
    plt.savefig(filepth, dpi=300)
    plt.close()

# Writes a plot of a ROCAUC to the log directory
def visualize_roc_curve(fpr, tpr, roc_auc, title='ROC Curve', tag=''):
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()

    rundir = utils.run_dir()

    filepth = os.path.join(rundir, f"ROC_{tag}_{time.strftime('%Y%m%d-%H%M%S')}.png")
    plt.savefig(filepth, dpi=300)
    plt.close()