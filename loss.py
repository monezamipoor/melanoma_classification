import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, opt):
        super(FocalLoss, self).__init__()
        self.alpha = 1.0
        self.gamma = 2.0
        self.reduction = 'mean'

        if opt['model']['focal_loss']['gamma'] > 0:
            self.gamma = opt['model']['focal_loss']['gamma']
        if opt['model']['focal_loss']['alpha'] > 0:
            self.alpha = opt['model']['focal_loss']['alpha']
        if opt['model']['focal_loss']['reduction'] is not None:
            self.reduction = opt['model']['focal_loss']['reduction']

        self.BCE_Func = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        preds = torch.sigmoid(inputs)           # Our predictions needs to be 0..1
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")    # Calc base BCE loss
        pt = preds * targets + (1 - preds) * (1 - targets)          # Predictions vs target labels
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss         # Apply gamma and alpha multipliers to predictions vs BCE loss

        # Apply reduction (mean, sum, or no reduction)
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss