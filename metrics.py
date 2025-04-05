import torch
from sklearn.metrics import confusion_matrix,  roc_curve, auc, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from torchmetrics.functional.classification import (
    binary_auroc,
    binary_accuracy,
    precision,
    recall,
    binary_f1_score,
    average_precision
)

def evaluate_metrics(opt, preds, target):
    results = {}
    
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
            results['AUC'] = round(to_scalar(binary_auroc(preds, target, thresholds=5)),4)
        elif metric_lower == 'accuracy':
            results['Accuracy'] = round(to_scalar(binary_accuracy(preds, target, threshold=0.5)),4)
        elif metric_lower == 'precision':
            results['Precision'] = round(to_scalar(precision(preds, target, threshold=0.5, task='binary')),4)
        elif metric_lower == 'recall':
            results['Recall'] = round(to_scalar(recall(preds, target, threshold=0.5, task='binary')),4)
        elif metric_lower in ['f1', 'f1 score']:
            results['F1 Score'] = round(to_scalar(binary_f1_score(preds, target, threshold=0.5)),4)
        elif metric_lower in ['average precision', 'ap']:
            results['Average Precision'] = round(to_scalar(average_precision(preds, target, threshold=0.5)),4)
        elif metric_lower == 'map':
            results['mAP'] = average_precision_score(target_int.numpy(), preds.numpy())
        elif metric_lower in ['confusion matrix', 'cm']:
            cm = confusion_matrix(target_int.numpy(), preds_binary.numpy())
            results['Confusion Matrix'] = cm
            visualize_confusion_matrix(cm)
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
