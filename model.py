import torch
import torch.nn as nn
import timm

class MelanomaModel(nn.Module):
    def __init__(self, opt):
        super(MelanomaModel, self).__init__()

        backbone_name = opt['model']['backbone']
        pretrained = opt['model']['pretrained']
        output_neurons = opt['model']['output_neurons']
        dropout_rate = opt['model']['dropout_rate']

        self.backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained,
            num_classes=0 
        )
        feature_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, output_neurons)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


def melanoma_model(opt):
    model = MelanomaModel(opt)
    return model


def melanoma_loss(opt):
    if 'focal_loss_gamma' in opt['model'] and opt['model']['focal_loss_gamma'] > 0:

        from torch.nn import functional as F
        
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
        return nn.CrossEntropyLoss()