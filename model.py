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
        self.num_unfrozen_layers = opt['model'].get('num_unfrozen_layers', None)

        # Check if using a transformer-based architecture
        self.is_transformer = any(keyword in backbone_name.lower() for keyword in ["swin", "vit"])

        if self.is_transformer:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            feature_dim = self.backbone.num_features
            self.classifier = nn.Sequential(

                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, opt['model']['output_neurons'])  # Output shape: [B, 1]
            )

        elif "efficientnet" in backbone_name.lower():
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained)

            # EfficientNet uses .classifier as its final layer

            feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, opt['model']['output_neurons'])
            )
            self.classifier = None
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained)

            if hasattr(self.backbone, 'fc'):
                feature_dim = self.backbone.num_features
                if hasattr(self.backbone, 'global_pool'):
                    self.backbone.global_pool = nn.AdaptiveAvgPool2d(1)
                self.backbone.fc = nn.Sequential(
                    nn.Flatten(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(feature_dim, opt['model']['output_neurons'])
                )
                self.classifier = None
            elif hasattr(self.backbone, 'head'):
                feature_dim = self.backbone.head.in_features
                self.backbone.head = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(feature_dim, opt['model']['output_neurons'])
                )
                self.classifier = None
            elif hasattr(self.backbone, 'classifier'):
                feature_dim = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Dropout(dropout_rate),
                    nn.Linear(feature_dim, opt['model']['output_neurons'])
                )
                self.classifier = None
            else:
                raise ValueError("Unsupported backbone structure: no classifier head found.")

        if self.freeze_backbone:
            self.freeze_layers()

    def freeze_layers(self):
        if self.is_transformer:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self._set_module_trainable(self.classifier, True)
            print("Transformer backbone frozen; classifier remains trainable.")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False

            # Detect classifier
            classifier = getattr(self.backbone, 'fc', None) or \
                         getattr(self.backbone, 'head', None) or \
                         getattr(self.backbone, 'classifier', None)

            if classifier:
                self._set_module_trainable(classifier, True)

            children = list(self.backbone.children())
            if classifier in children:
                children.remove(classifier)

            if self.num_unfrozen_layers and len(children) > 0:
                to_unfreeze = children[-self.num_unfrozen_layers:]
                for module in to_unfreeze:
                    self._set_module_trainable(module, True)
                print(f"Unfroze the last {self.num_unfrozen_layers} modules + classifier.")
            else:
                print("Only classifier head remains trainable.")

    @staticmethod
    def _set_module_trainable(module, trainable):
        for param in module.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        if self.is_transformer:
            features = self.backbone.forward_features(x)  # shape: [B, H, W, C]

            if features.ndim == 4:
                # [B, H, W, C] → [B, C, H, W] for pooling
                features = features.permute(0, 3, 1, 2)         # [B, C, H, W]
                features = torch.nn.functional.adaptive_avg_pool2d(features, 1)  # [B, C, 1, 1]
                features = features.view(features.size(0), -1)  # [B, C]
            elif features.ndim == 3:
                features = features[:, 0]  # [B, D]
            elif features.ndim == 2:
                pass  # Already [B, D]
            else:
                raise ValueError(f"Unexpected shape: {features.shape}")

            return self.classifier(features).squeeze(-1)  # [B]
        else:
            return self.backbone(x).squeeze(-1)


def test_melanoma_model(opt, testmodel):

    model = MelanomaModel(opt)

    if testmodel is not None:
        print('Loading saved model: ', testmodel)
        model.load_state_dict(torch.load(testmodel))

    return model

def train_melanoma_model(opt):
    model = MelanomaModel(opt)
    return model
