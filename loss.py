"""
In this module we calculate the loss functions for the model.
The loss functions are defined in the following order:
1. Focal Loss
2. Dice Loss
3. SVM Hinge Loss
4. Supervised Contrastive Loss
5. Combined Loss
6. Hard Triplet Loss
   
"""


import torch
import torch.nn as nn
from torch import nn as nn
import torch.nn.functional as F
from utils import check_nested_key

class FocalLoss(nn.Module):
    """
    Focal loss is a loss function designed to address class imbalance by down-weighting easy examples and focusing on hard examples.
    
    """
    def __init__(self, opt):
        super(FocalLoss, self).__init__()
        # Focal loss parameters
        self.alpha = 1.0
        self.gamma = 2.0
        self.reduction = 'mean'

        # Check focal loss parameters in the config
        if opt['model']['focal_loss']['gamma'] > 0:
            self.gamma = opt['model']['focal_loss']['gamma']
        if opt['model']['focal_loss']['alpha'] > 0:
            self.alpha = opt['model']['focal_loss']['alpha']
        if opt['model']['focal_loss']['reduction'] is not None:
            self.reduction = opt['model']['focal_loss']['reduction']

        print("Focal Loss Alpha/Gamma/Reduction:", self.alpha,'/' ,self.gamma,'/',self.reduction)
        # Initialize the BCE loss function
        # Note: We use BCEWithLogitsLoss to combine sigmoid and binary cross-entropy in one step
        self.BCE_Func = nn.BCEWithLogitsLoss(reduction='none')


    def forward(self, inputs, targets):
        # First, we need to calculate the binary cross-entropy loss
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        # Then, we calculate the probabilities using sigmoid
        probs = torch.sigmoid(inputs)
        # Calculate the focal loss components
        pt = probs * targets + (1 - probs) * (1 - targets)
        
        # If alpha is scalar, this is enough:
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        # Calculate the focal loss by combining the components. The formula is implemented as the deinition of focal loss.
        loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            # If reduction is mean, we need to average the loss
            return loss.mean()
        elif self.reduction == 'sum':
            # If reduction is sum, we need to sum the loss
            return loss.sum()
        return loss

class DiceLoss(nn.Module):
    """
    Dice loss is a loss function used for image segmentation tasks, particularly in medical imaging.
    It is based on the Dice coefficient, which measures the overlap between two sets.
    """
    def __init__(self, opt):
        # Initialize the Dice loss with a smoothing factor to avoid division by zero
        super(DiceLoss, self).__init__()
        # Get the smoothing factor from the config
        self.smooth = opt['model'].get('dice_loss_smooth', 1e-6)
        print("Dice loss smoothing:", self.smooth)

    def forward(self, logits, targets):
        # Apply sigmoid to logits
        probs = torch.sigmoid(logits)
        # Convert targets to float
        targets = targets.float() 

        # Flatten the tensors
        probs = probs.view(-1)
        targets = targets.view(-1)

        # Calculate the intersection and union
        intersection = (probs * targets).sum()
        # Calculate the Dice score
        dice_score = (2. * intersection + self.smooth) / (probs.sum() + targets.sum() + self.smooth)

        return 1 - dice_score  # Dice loss

class SVMHingeLoss(nn.Module):
    """
    SVM Hinge Loss is a loss function used in Support Vector Machines (SVMs) for binary classification tasks.

    """
    def __init__(self):
        # Initialize the SVM Hinge Loss
        super(SVMHingeLoss, self).__init__()

    def forward(self, outputs, targets):
        # Convert targets to float and map {0,1} to {-1,1}
        targets = targets.float() * 2 - 1  
        # Calculate the hinge loss
        loss = torch.clamp(1 - outputs * targets, min=0)
        return loss.mean()

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss is a loss function used for contrastive learning tasks.
    It encourages similar samples to be close in the embedding space and dissimilar samples to be far apart.
    
    """
    def __init__(self, temperature=0.1, margin_threshold=0.5):
        # Initialize the Supervised Contrastive Loss
        # temperature: controls the scale of the logits
        super(SupConLoss, self).__init__()
        # Set the temperature and margin threshold
        self.temperature = temperature
        self.margin_threshold = margin_threshold

    def forward(self, features, labels):
        # Pass the features to the device
        device = features.device
        # Use the batch size from the features
        batch_size = features.shape[0]

        # Reshape the labels to match the batch size
        labels = labels.contiguous().view(-1, 1) 
        # Create a mask for positive pairs
        # The mask is 1 for positive pairs and 0 for negative pairs
        mask = torch.eq(labels, labels.T).float().to(device) 
        # Calculate the anchor dot contrast
        # This is the cosine similarity between the features
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        # Set the diagonal to a large negative value to ignore self-similarity
        # This is done to avoid self-similarity in the contrastive loss
        logits_mask = torch.ones_like(anchor_dot_contrast) - torch.eye(batch_size, device=device)
        anchor_dot_contrast = anchor_dot_contrast * logits_mask
        # Set the diagonal to a large negative value to ignore self-similarity
        with torch.no_grad():
            # calculate the cosine similarity after normalization
            cosine_sim = torch.matmul(F.normalize(features, dim=1), F.normalize(features, dim=1).T)
            # calculate the hard negative mask by checking if the cosine similarity is greater than the margin threshold and not equal to 1
            # The hard negative mask is 1 for hard negatives and 0 for easy negatives
            hard_neg_mask = (1 - mask) * (cosine_sim > self.margin_threshold).float()

        # Apply the mask to the logits
        # The logits are the cosine similarity between the features
        denom_mask = (mask + hard_neg_mask) * logits_mask

        # Calculate the exponential of the logits
        # The logits are the cosine similarity between the features
        exp_logits = torch.exp(anchor_dot_contrast) * denom_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)

        # Only positive pairs in numerator
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)

        return -mean_log_prob_pos.mean()

class CombinedLoss(nn.Module):
    """
    Combined loss function that combines two different loss functions.
    
    """
    def __init__(self, opt):
        super().__init__()
        # Get the loss combination weight from the config
        self.weight = opt['model'].get('loss_combination_weight', 1.0)
        # Get the first and second loss function names from the config
        self.first_name = opt['model']['loss_function']
        # Get the second loss function name from the config
        self.second_name = opt['model'].get('second_loss', '')
        # Call two loss functions
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
            # If the second loss is triplet or contrastive, we need to use the features
            if features is None:
                raise ValueError("Features required for feature-based loss but not provided.")
            # If the second loss is triplet or contrastive, we need to use the features
            loss2 = self.l2(features, targets)
        else:
            # If the second loss is not triplet or contrastive, we need to use the logits
            loss2 = self.l2(logits, targets)
        # Return the combined loss
        # The combined loss is the sum of the first loss and the second loss multiplied by the weight   
        return loss1 + self.weight * loss2

class HardTripletLoss(nn.Module):
    """
    Hard Triplet Loss is a loss function used in triplet networks for metric learning tasks.
    It encourages the model to learn a distance metric that separates positive and negative samples.

    """
    def __init__(self, margin: float = 1.0):
        # Initialize the Hard Triplet Loss
        super().__init__()
        # Identify the Triplet Margin Loss
        self.base = nn.TripletMarginLoss(margin=margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        # Take anchors, positives, and negatives from the embeddings. 
        # The embeddings are the output of the model
        # The labels are the ground truth labels for the samples
        a, p, n = batch_hard_triplet_embeddings(embeddings, labels)
        # Calculate the triplet loss using the base TripletMarginLoss
        return self.base(a, p, n)
    
# A helper to create the loss, it is useful for adding combined losses alongside single losses. 
def _make_loss(name, opt, loader=None):
    """
    Returns the desired loss module by name, including 'combined'.
    """
    # convert the name to lowercase
    name = name.lower()
    if name == 'bce':
        # Check if the model has a BCE loss weight
        # If the model has a BCE loss weight, use it
        # If the model does not have a BCE loss weight, use the default value of 1.0
        weights = opt['model'].get('bce_loss_weights', [1.0])
        print(f"Using BCE Loss with weights: {weights}")
        # Calculate the BCE loss with logits
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights))

    if name == 'dice':
        # Check if the model has a Dice loss smooth factor
        print("Using Dice Loss (smooth={})".format(opt['model'].get('dice_loss_smooth')))
        return DiceLoss(opt)

    if name == 'focal':
        # Check if the model has a Focal loss configuration
        fl_cfg = opt['model']['focal_loss']
        print(f"Using Focal Loss (α={fl_cfg['alpha']}, γ={fl_cfg['gamma']}, reduction={fl_cfg['reduction']})")
        # Check if the model has a Focal loss weight
        # return the Focal loss with the specified parameters
        return FocalLoss(opt)

    if name == 'svm_hinge':
        # Check if the model has a SVM Hinge loss configuration
        print("Using SVM Hinge Loss")
        # return the SVM Hinge loss
        return SVMHingeLoss()

    if name == 'contrastive':
        # Check if the model has a Contrastive loss configuration
        # If the model has a Contrastive loss configuration, use it
        # If the model does not have a Contrastive loss configuration, use the default value of 0.07
        temp   = opt['model'].get('contrastive_temperature', 0.07)
        # Check if the model has a Contrastive loss margin
        # If the model has a Contrastive loss margin, use it
        margin = opt['model'].get('contrastive_margin',    0.5)
        print(f"Using Supervised Contrastive Loss (temp={temp}, margin={margin})")
        # return the Contrastive loss with the specified parameters
        return SupConLoss(temperature=temp, margin_threshold=margin)
    
    if name == 'triplet':
        # Check if the model has a Triplet loss configuration
        # If the model has a Triplet loss configuration, use it
        margin = opt['model'].get('triplet_margin', 1.0)
        print(f"Using Triplet Margin Loss (margin={margin})")
        # return the Triplet loss with the specified parameters
        return HardTripletLoss(margin)

    if name == 'bce_auto' and loader is not None:
        # Check if the model has a BCE loss configuration
        print("Using bce_auto Loss")
        # create a list to store all labels
        all_labels = []
        # inputs, labels with the inputs being omitted
        for _, labels in loader:
            # move the labels to the CPU
            all_labels.append(labels.cpu())
        # concatenate all labels into a single tensor
        all_labels = torch.cat(all_labels)

        num_pos = all_labels.sum()
        # calculate the number of positive samples
        num_neg = len(all_labels) - num_pos
        # calculate the number of negative samples
        # calculate the positive weight
        pos_weight = torch.tensor([num_neg / num_pos])

        print("Using Auto-weighted BCE")
        print(f"  Num pos {num_pos}, neg {num_neg}, ratio {num_neg/num_pos:.3f}")

        # return the BCE loss with the specified parameters
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # If we got here just default to BCE
    bce_weights = opt['model'].get('bce_loss_weights', [1.0])
    print("Defaulting using BCE Loss with weights:", bce_weights)
    # return the BCE loss with the specified parameters
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(bce_weights))

def batch_hard_triplet_embeddings(embeddings: torch.Tensor, labels: torch.Tensor):
    """
    Given a batch of embeddings and their corresponding labels, this function computes the hardest positive and negative samples
    
    """
    # Calculate the pairwise distance between all embeddings
    # The pairwise distance is calculated using the Euclidean distance
    dist = torch.cdist(embeddings, embeddings, p=2)
    # Create a mask for positive and negative pairs
    # The mask is 1 for positive pairs and 0 for negative pairs
    mask_pos = labels.unsqueeze(1) == labels.unsqueeze(0)
    mask_neg = labels.unsqueeze(1) != labels.unsqueeze(0)
    # Remove self-comparisons from the positive mask
    diag = torch.eye(labels.size(0), device=labels.device, dtype=torch.bool)
    mask_pos = mask_pos & ~diag
    #Select the hardest positive for each anchor
    pos_dist = dist.clone()
    pos_dist[~mask_pos] = float('-inf')
    hardest_pos = torch.argmax(pos_dist, dim=1)
    # Set all non-positive distances to -inf so they don't affect the argmax.
    neg_dist = dist.clone()
    neg_dist[~mask_neg] = float('inf')
    # Set non-negatives to inf so they don't affect the argmin.
    hardest_neg = torch.argmin(neg_dist, dim=1)
    anchor = embeddings
    positive = embeddings[hardest_pos]
    negative = embeddings[hardest_neg]
    # Return the anchor, positive, and negative samples
    return anchor, positive, negative


def melanoma_loss(opt, loader=None):
    """
    This function returns the loss function based on the configuration provided in the opt dictionary.
    """
    # Check if the model has a combined loss configuration
    if opt['model'].get('combined_loss', False):
        print("Building CombinedLoss from "
                f"{opt['model']['loss_function']} + "
                f"{opt['model']['second_loss']} * "
                f"{opt['model']['loss_combination_weight']}")
        return CombinedLoss(opt)
    # If the model does not have a combined loss configuration, return the specified loss function
    loss_name = opt['model'].get('loss_function', 'bce')
    return _make_loss(loss_name, opt, loader)
