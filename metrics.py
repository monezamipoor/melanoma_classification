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
from wandb_helper import wandb_log_cm


def find_best_threshold(y_true, y_probs):
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = f1.argmax()
    return thresholds[best_idx], precision[best_idx], recall[best_idx], f1[best_idx]


def evaluate_metrics(opt, preds, target, epoch):

    results = {}
    classlabels = ["Benign", "Malignant"]

    # Convert probabilities to binary predictions using a threshold of 0.5.
    preds_binary = (preds > 0.5).int()
    target_int = target.int()

    # Retrieve the list of metrics to compute from your configuration.
    config_metrics = opt['testing']['model_metrics']
    def to_scalar(x):
        return x.item() if isinstance(x, torch.Tensor) else x

    # Loop over each metric name and compute the corresponding metric.
    for metric in config_metrics:
        metric_lower = metric.lower()
        if metric_lower == 'auc':
            results['AUC'] = round(to_scalar(binary_auroc(preds, target, thresholds=None)),4)
        elif metric_lower == 'accuracy':
            results['Accuracy'] = round(to_scalar(binary_accuracy(preds, target, threshold=0.5)),4)
        elif metric_lower == 'precision':
            results['Precision'] = round(to_scalar(binary_precision(preds, target, threshold=0.5)),4)
        elif metric_lower == 'recall':
            results['Recall'] = round(to_scalar(binary_recall(preds, target, threshold=0.5)),4)
        elif metric_lower in ['f1', 'f1 score']:
            results['F1 Score'] = round(to_scalar(binary_f1_score(preds, target, threshold=0.5)),4)
        elif metric_lower in ['average precision', 'ap']:
            results['Average Precision'] = round(to_scalar(binary_average_precision(preds, target)),4)
        elif metric_lower == 'map':
            results['mAP'] = average_precision_score(target_int.numpy(), preds.numpy())
        elif metric_lower in ['confusion matrix', 'cm']:
            cm = confusion_matrix(target_int.numpy(), preds_binary.numpy())
            results["Confusion Matrix - Epoch: " + str(epoch)] = cm

            # TODO break out wandb and local display of CMs. Also save CM as part of logging rather than plot them
            wandb_log_cm(preds_binary.cpu().numpy().flatten().tolist(), target_int.cpu().numpy().flatten().tolist(), classlabels, "Confusion Matrix - Epoch: " + str(epoch))
            #visualize_confusion_matrix(cm, classlabels, "Confusion Matrix - Epoch: " + str(epoch))
        elif metric_lower == 'roc':
            fpr, tpr, roc_thresholds = roc_curve(target_int.numpy(), preds.numpy())
            roc_auc_val = auc(fpr, tpr)
            results['ROC'] = {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': roc_thresholds,
                'roc_auc': roc_auc_val
            }
            visualize_roc_curve(fpr, tpr, roc_auc_val)
    
    return results

def visualize_confusion_matrix(cm, labels=['Negative', 'Positive'], title='Confusion Matrix'):
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels,
                yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def visualize_roc_curve(fpr, tpr, roc_auc, title='ROC Curve'):
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
    plt.show()
