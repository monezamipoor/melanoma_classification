import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class MelanomaModel(nn.Module):
    def __init__(self, opt):
        super(MelanomaModel, self).__init__()

        backbone_name = opt['model']['backbone']
        pretrained = opt['model']['pretrained']
        #output_neurons = opt['model']['output_neurons']
        dropout_rate = opt['model']['dropout_rate']

        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained
        )
        feature_dim = self.backbone.num_features

        # Custom head
        self.backbone.global_pool = nn.AdaptiveAvgPool2d(1)  # Average mean pooling
        self.backbone.fc = nn.Sequential(
            nn.Flatten(),           # Reduce feature set
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 1),  # Binary classification output
        )
    
    def forward(self, x):
        return self.backbone(x)


def melanoma_model(opt):
    model = MelanomaModel(opt)
    return model

# TODO Focal Loss not working
def melanoma_loss(opt):
    if 'focal_loss_gamma' in opt['model'] and opt['model']['focal_loss_gamma'] > 0:

        class FocalLoss(nn.Module):
            def __init__(self, gamma=2.0):
                super(FocalLoss, self).__init__()
                self.gamma = gamma

            def forward(self, input, target):
                ce_loss = F.cross_entropy(input, target, reduction='none')
                pt = torch.exp(-ce_loss)
                loss = (1 - pt) ** self.gamma * ce_loss
                return loss.mean()

        return FocalLoss(gamma=opt['model']['focal_loss_gamma'])
    else:
        return nn.BCEWithLogitsLoss()               # Was CE. Now BCE because its a 0 OR 1 problem, not 0 AND 1.