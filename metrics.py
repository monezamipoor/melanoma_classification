import torch
import torchmetrics
from torchmetrics.functional.classification import binary_auroc

def evaluate_metrics(opt, all_outputs, all_labels):
    # Skeleton function for evaluating metrics based on predictions and ground truths
    # Implement evaluation logic here using libraries like sklearn or torchmetrics
    # metrics is a dictionary like {'mAP' : 0.5, 'recall' : 0.9}
    metrics = {}

    #TODO Implement metrics
    metrics_list = opt['testing']['model_save_metric']

    for index, value in enumerate(metrics_list):
        if value == 'AUC':
            metrics['AUC'] = BinaryAUC(all_outputs, all_labels).item()

    return metrics

# Binary AUC metric
def BinaryAUC(preds, target):
    auc = binary_auroc(preds, target, thresholds=5)
    print(auc)
    return auc