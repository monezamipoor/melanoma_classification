import torch
import torch.nn as nn
from torch import nn as nn
import torch.nn.functional as F
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

class SVMHingeLoss(nn.Module):
    def __init__(self):
        super(SVMHingeLoss, self).__init__()

    def forward(self, outputs, targets):
        targets = targets.float() * 2 - 1  # Convert {0,1} --> {-1,1}
        loss = torch.clamp(1 - outputs * targets, min=0)
        return loss.mean()

class SupConLoss(nn.Module):
    """Supervised Contrastive Loss with Dynamic Hard Negative Mining."""
    def __init__(self, temperature=0.1, margin_threshold=0.5):
        """
        Args:
            temperature (float): Temperature scaling for softmax.
            margin_threshold (float): Cosine similarity threshold to select hard negatives.
        """
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.margin_threshold = margin_threshold
        print(f"Using Supervised Contrastive Loss with temp={self.temperature}, margin={self.margin_threshold}")

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)  # [2N, 1]
        mask = torch.eq(labels, labels.T).float().to(device)  # [2N, 2N]  Positive pairs

        # --- Compute cosine similarity matrix ---
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )  # [2N, 2N]

        # --- Remove self-contrast cases (diagonal) ---
        logits_mask = torch.ones_like(anchor_dot_contrast) - torch.eye(batch_size, device=device)
        anchor_dot_contrast = anchor_dot_contrast * logits_mask

        # --- Dynamic Hard Negative Mining ---
        # Compute cosine similarities (without temperature scaling)
        cosine_sim = torch.matmul(F.normalize(features, dim=1), F.normalize(features, dim=1).T)

        # Hard negatives: different label, but cosine similarity > margin
        neg_mask = (1 - mask) * (cosine_sim > self.margin_threshold).float()

        # --- Combine positive and selected negatives into final mask ---
        combined_mask = mask + neg_mask

        # --- Compute log_prob ---
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Only positive pairs + hard negatives contribute
        mean_log_prob_pos = (combined_mask * log_prob).sum(1) / (combined_mask.sum(1) + 1e-9)

        # --- Final loss ---
        loss = -mean_log_prob_pos
        loss = loss.mean()

        return loss


def melanoma_loss(opt):

    loss_function = opt['model'].get('loss_function', 'bce')

    if loss_function == 'focal':
        print("Using Focal Loss")
        return FocalLoss(opt)
    elif loss_function == 'dice':
        print("Using Dice Loss")
        return DiceLoss(opt)
    elif loss_function == 'svm_hinge':
        print("Using SVM Hinge Loss")
        return SVMHingeLoss()
    elif loss_function == 'contrastive':
        return SupConLoss(
            temperature=opt['model'].get('contrastive_temperature', 0.07),
            margin_threshold=opt['model'].get('contrastive_margin', 0.5)  # 👈 Add this to your YAML later!
        )
    # If we got here then just use BCE

    bce_weights = opt['model'].get('bce_loss_weights', [1.0])
    print("Using BCE Loss with weights:", bce_weights)
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bce_weights))
