import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

# class MelanomaModel(nn.Module):
#     def __init__(self, opt):
#         super(MelanomaModel, self).__init__()

#         backbone_name = opt['model']['backbone']
#         pretrained = opt['model']['pretrained']
#         #output_neurons = opt['model']['output_neurons']
#         dropout_rate = opt['model']['dropout_rate']

#         self.backbone = timm.create_model(
#             backbone_name, 
#             pretrained=pretrained
#         )
#         feature_dim = self.backbone.num_features

#         # Custom head
#         self.backbone.global_pool = nn.AdaptiveAvgPool2d(1)  # Average mean pooling
#         self.backbone.fc = nn.Sequential(
#             nn.Flatten(),           # Reduce feature set
#             nn.Dropout(dropout_rate),
#             nn.Linear(feature_dim, 1),  # Binary classification output
#         )
    
#     def forward(self, x):
#         return self.backbone(x)

import torch
import torch.nn as nn
import timm

class MelanomaModel(nn.Module):
    def __init__(self, opt):
        super(MelanomaModel, self).__init__()

        backbone_name = opt['model']['backbone']
        pretrained = opt['model']['pretrained']
        dropout_rate = opt['model']['dropout_rate']
        self.freeze_backbone = opt['model'].get('freeze_backbone', False)
        
        self.num_frozen_layers = opt['model'].get('num_frozen_layers', None)


        self.backbone = timm.create_model(backbone_name, pretrained=pretrained)

        if hasattr(self.backbone, 'fc'):
            feature_dim = self.backbone.num_features
            if hasattr(self.backbone, 'global_pool'):
                self.backbone.global_pool = nn.AdaptiveAvgPool2d(1)
            self.backbone.fc = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, 1) 
            )
        elif hasattr(self.backbone, 'head'):
            feature_dim = self.backbone.head.in_features
            self.backbone.head = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, 1)
            )
        elif hasattr(self.backbone, 'classifier'):
            feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, 1)
            )
        else:
            raise ValueError("The backbone model does not have a recognized classifier head attribute.")

        if self.freeze_backbone:
            self.freeze_layers()

    def freeze_layers(self):
        """
        Freeze layers dynamically based on self.num_frozen_layers.
        If self.num_frozen_layers is specified, freeze all backbone modules
        except the last self.num_frozen_layers.
        Otherwise, freeze all parameters except those belonging to the classifier head.
        """
        if self.num_frozen_layers is not None:
            # Get an ordered list of backbone children.
            children = list(self.backbone.children())
            num_children = len(children)
            # We'll unfreeze the last `self.num_frozen_layers` modules.
            start_unfreeze = max(num_children - self.num_frozen_layers, 0)
            for idx, child in enumerate(children):
                if idx < start_unfreeze:
                    for param in child.parameters():
                        param.requires_grad = False
                else:
                    for param in child.parameters():
                        param.requires_grad = True
            print(f"Frozen the first {start_unfreeze} modules; last {self.num_frozen_layers} modules are trainable.")
        else:
            # Default behavior: freeze all layers except those in classifier head.
            for name, param in self.backbone.named_parameters():
                if "fc" in name or "head" in name or "classifier" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print("Backbone layers have been frozen except classifier head.")

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