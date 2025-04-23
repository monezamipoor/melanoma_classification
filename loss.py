import torch
import torch.nn as nn
from torch import nn as nn

from utils import check_nested_key

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

        print("Focal Loss Alpha/Gamma/Reduction:", self.alpha,'/' ,self.gamma,'/',self.reduction)

        self.BCE_Func = nn.BCEWithLogitsLoss(reduction='none')


    def forward(self, inputs, targets):
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        probs = torch.sigmoid(inputs)
        pt = probs * targets + (1 - probs) * (1 - targets)
        
        # If alpha is scalar, this is enough:
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


#     def forward(self, inputs, targets):
#         preds = torch.sigmoid(inputs)           # Our predictions needs to be 0..1
#         BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")    # Calc base BCE loss
#         pt = preds * targets + (1 - preds) * (1 - targets)          # Predictions vs target labels
#         focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss         # Apply gamma and alpha multipliers to predictions vs BCE loss

#         # Apply reduction (mean, sum, or no reduction)
#         if self.reduction == 'mean':
#             return focal_loss.mean()
#         elif self.reduction == 'sum':
#             return focal_loss.sum()
#         else:
#             return focal_loss


class DiceLoss(nn.Module):
    def __init__(self, opt):
        super(DiceLoss, self).__init__()
        self.smooth = opt['model'].get('dice_loss_smooth', 1e-6)
        print("Dice loss smoothing:", self.smooth)

    def forward(self, logits, targets):
        # Apply sigmoid to logits
        probs = torch.sigmoid(logits)
        targets = targets.float()  # Ensure float

        # Flatten the tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice_score = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice_score  # Dice loss


def melanoma_loss(opt, loader=None):

    loss_function = opt['model'].get('loss_function', 'bce')

    if loss_function == 'focal':
        print("Using Focal Loss")
        return FocalLoss(opt)
    elif loss_function == 'dice':
        print("Using Dice Loss")
        return DiceLoss(opt)
    elif loss_function == 'bce_auto' and loader is not None:
        # Calculates the "perfect" weighted loss for BCE based on the loader class label balance
        all_labels = []
        for inputs, labels in loader:
            # Move labels to CPU if necessary
            labels = labels.cpu()
            all_labels.append(labels)

        # Concatenate all labels into a single tensor
        all_labels = torch.cat(all_labels)

        print('Using Auto-weigted BCE')
        num_pos = all_labels.sum()
        num_neg = len(all_labels) - num_pos

        print ("Num Pos, Neg, Ratio:", str(num_pos), ', ', str(num_neg), ',',  str(num_neg / num_pos))

        pos_weight = torch.tensor([num_neg / num_pos])
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # If we got here then just use BCE

    bce_weights = opt['model'].get('bce_loss_weights', [1.0])
    print("Using BCE Loss with weights:", bce_weights)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bce_weights))