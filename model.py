import torch.nn as nn
import timm
from loss import FocalLoss
from utils import check_nested_key

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

# Comment out the focal loss lines in config to default to BCE Loss
def melanoma_loss(opt):

    if check_nested_key (opt, ['model', 'focal_loss']):
        if opt['model']['focal_loss']['gamma'] > 0 and opt['model']['focal_loss']['alpha'] > 0:
            print("Using Focal Loss")
            return FocalLoss(opt)
    else:
        print("Using BCE Loss")
        return nn.BCEWithLogitsLoss()               # Was CE. Now BCE because its a 0 OR 1 problem, not 0 AND 1.


