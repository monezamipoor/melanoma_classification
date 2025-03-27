import torch
import torchmetrics
from torchmetrics.functional import matthews_corrcoef
from sklearn.metrics import confusion_matrix, average_precision_score, average_precision_score
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

def evaluate_metrics(opt, preds, target, metric):
    # Skeleton function for evaluating metrics based on predictions and ground truths
    # Implement evaluation logic here using libraries like sklearn or torchmetrics
    # metrics is a dictionary like {'mAP' : 0.5, 'recall' : 0.9}
    metrics = {}
    
    # Convert probabilities to binary predictions using a threshold of 0.5
    preds_binary = (preds > 0.5).int()
    target_int = target.int()
    

    #TODO Implement metrics
    metrics_list = opt['testing']['model_save_metric']

    for index, value in enumerate(metrics_list):
        
        if 'AUC' in metrics_to_compute:
            auc = binary_auroc(preds, target, thresholds=5)
            metrics['AUC'] = auc

        if metric == 'Accuracy':
            acc = binary_accuracy(preds, target, threshold=0.5)
            metrics['Accuracy'] = acc

        if metric == 'Precision':
            prec = precision(preds, target, threshold=0.5)
            metrics['Precision'] = prec

        if metric == 'Recall':
            rec = recall(preds, target, threshold=0.5)
            metrics['Recall'] = rec

        if metric == 'F1 Score':
            f1 = binary_f1_score(preds, target, threshold=0.5)
            metrics['F1 Score'] = f1

        if metric == 'Average Precision':
            avg_prec = average_precision(preds, target, threshold=0.5)
            metrics['Average Precision'] = avg_prec

        if metric == 'mAP':
            mAP = average_precision_score(target_int.numpy(), preds.numpy())
            metrics['mAP'] = mAP

        if metric == 'Confusion Matrix':
            cm = confusion_matrix(preds_binary.numpy(), target_int.numpy())
            metrics['Confusion Matrix'] = cm
            visualize_confusion_matrix(cm)

        if value == 'ROC':
            fpr, tpr, roc_thresholds = roc_curve(target_int.numpy(), preds.numpy())
            roc_auc = auc(fpr, tpr)  # AUC is area Under the ROC Curve
            metrics['ROC'] = {
                'fpr': fpr,          # False Positive Rate
                'tpr': tpr,          # True Positive Rate
                'thresholds': roc_thresholds, 
                'roc_auc': roc_auc
            }
            visualize_roc_curve(fpr, tpr, roc_auc)

    return metrics


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


# Binary AUC metric
def BinaryAUC(preds, target):
    auc = binary_auroc(preds, target, thresholds=5)
    print(auc)
    return auc
