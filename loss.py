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
    def __init__(self, temperature=0.1, margin_threshold=0.5):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.margin_threshold = margin_threshold

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)  # [2N, 1]
        mask = torch.eq(labels, labels.T).float().to(device)  # [2N, 2N] Positive mask

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        logits_mask = torch.ones_like(anchor_dot_contrast) - torch.eye(batch_size, device=device)
        anchor_dot_contrast = anchor_dot_contrast * logits_mask
        with torch.no_grad():
            cosine_sim = torch.matmul(F.normalize(features, dim=1), F.normalize(features, dim=1).T)
            hard_neg_mask = (1 - mask) * (cosine_sim > self.margin_threshold).float()

        denom_mask = (mask + hard_neg_mask) * logits_mask


        exp_logits = torch.exp(anchor_dot_contrast) * denom_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Only positive pairs in numerator
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        return -mean_log_prob_pos.mean()

class CombinedLoss(nn.Module):
    """L = L1 + weight * L2"""
    def __init__(self, opt):
        super().__init__()
        self.weight = opt['model'].get('loss_combination_weight', 1.0)
        self.first_name = opt['model']['loss_function']
        self.second_name = opt['model'].get('second_loss', '')
        self.l1 = _make_loss(self.first_name, opt)
        self.l2 = _make_loss(self.second_name, opt)
        print(f"Using CombinedLoss: {self.first_name} + {self.weight}*{self.second_name}")

    def forward(self, inputs, targets):
        # Unpack features and logits if model returns a tuple
        if isinstance(inputs, tuple) and len(inputs) == 2:
            features, logits = inputs
        else:
            features = None
            logits = inputs

        # First loss always on logits
        loss1 = self.l1(logits, targets)
        # Second loss: feature-based for triplet/contrastive
        if self.second_name.lower() in ['triplet', 'contrastive']:
            if features is None:
                raise ValueError("Features required for feature-based loss but not provided.")
            loss2 = self.l2(features, targets)
        else:
            loss2 = self.l2(logits, targets)

        return loss1 + self.weight * loss2

# A helper to create the loss, it is useful for adding combined losses alongside single losses. 
def _make_loss(name, opt, loader=None):
    """
    Returns the desired loss module by name, including 'combined'.
    """
    name = name.lower()
    if name == 'bce':
        weights = opt['model'].get('bce_loss_weights', [1.0])
        print(f"Using BCE Loss with weights: {weights}")
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights))

    if name == 'dice':
        print("Using Dice Loss (smooth={})".format(opt['model'].get('dice_loss_smooth')))
        return DiceLoss(opt)

    if name == 'focal':
        fl_cfg = opt['model']['focal_loss']
        print(f"Using Focal Loss (α={fl_cfg['alpha']}, γ={fl_cfg['gamma']}, reduction={fl_cfg['reduction']})")
        return FocalLoss(opt)

    if name == 'svm_hinge':
        print("Using SVM Hinge Loss")
        return SVMHingeLoss()

    if name == 'contrastive':
        temp   = opt['model'].get('contrastive_temperature', 0.07)
        margin = opt['model'].get('contrastive_margin',    0.5)
        print(f"Using Supervised Contrastive Loss (temp={temp}, margin={margin})")
        return SupConLoss(temperature=temp, margin_threshold=margin)
    
    if name == 'triplet':
        margin = opt['model'].get('triplet_margin', 1.0)
        print(f"Using Triplet Margin Loss (margin={margin})")
        return nn.TripletMarginLoss(margin=margin)

    if name == 'bce_auto' and loader is not None:
        print("Using bce_auto Loss")
        all_labels = []
        # inputs, labels with the inputs being omitted
        for _, labels in loader:
            all_labels.append(labels.cpu())
        all_labels = torch.cat(all_labels)

        num_pos = all_labels.sum()
        num_neg = len(all_labels) - num_pos
        pos_weight = torch.tensor([num_neg / num_pos])

        print("Using Auto-weighted BCE")
        print(f"  Num pos {num_pos}, neg {num_neg}, ratio {num_neg/num_pos:.3f}")
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    raise ValueError(f"Unknown loss function: {name}")

def melanoma_loss(opt, loader=None):
    if opt['model'].get('combined_loss', False):
        print("Building CombinedLoss from "
                f"{opt['model']['loss_function']} + "
                f"{opt['model']['second_loss']} * "
                f"{opt['model']['loss_combination_weight']}")
        return CombinedLoss(opt)
    loss_name = opt['model'].get('loss_function', 'bce')
    return _make_loss(loss_name, opt, loader)