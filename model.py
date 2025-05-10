import torch
import torch.nn as nn
import timm

class MelanomaModel(nn.Module):
    def __init__(self, opt):
        super(MelanomaModel, self).__init__()

        self.opt = opt   
        mode = opt['model'].get('mode', 'regular').lower()

        if mode == 'regular':
            self.use_svm_head = False
            self.use_contrastive_svm = False
            self.use_contrastive_head = False
        elif mode == 'svm':
            self.use_svm_head = True
            self.use_contrastive_svm = False
            self.use_contrastive_head = False
        elif mode == 'contrastive':
            self.use_svm_head = False
            self.use_contrastive_svm = False
            self.use_contrastive_head = True
        elif mode == 'svm+contrastive':
            self.use_svm_head = True
            self.use_contrastive_svm = True
            self.use_contrastive_head = True
        else:
            raise ValueError(f"Unknown mode {mode}! Must be one of: regular, svm, contrastive, svm+contrastive")


        backbone_name = opt['model']['backbone']
        pretrained = opt['model']['pretrained']
        dropout_rate = opt['model']['dropout_rate']
        self.freeze_backbone = opt['model'].get('freeze_backbone', False)
        self.num_unfrozen_layers = opt['model'].get('num_unfrozen_layers', None)


        self.is_transformer = any(keyword in backbone_name.lower() for keyword in ["swin", "vit"])

        # --- Backbone ---
        if self.is_transformer:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            feature_dim = self.backbone.num_features
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, opt['model']['output_neurons'])
            )
        else:
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
            if hasattr(self.backbone, 'fc'):
                feature_dim = self.backbone.fc.in_features
                if hasattr(self.backbone, 'global_pool'):
                    self.backbone.global_pool = nn.Identity()  
                self.backbone.fc = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(feature_dim, opt['model']['output_neurons'])
                )
                self.classifier = self.backbone.fc
            elif hasattr(self.backbone, 'head'):
                feature_dim = self.backbone.head.in_features
                self.backbone.head = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(feature_dim, opt['model']['output_neurons'])
                )
                self.classifier = self.backbone.head
            elif hasattr(self.backbone, 'classifier'):
                feature_dim = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(feature_dim, opt['model']['output_neurons'])
                )
                self.classifier = self.backbone.classifier

            else:
                raise ValueError("Unsupported backbone structure: no classifier head found.")
        
        # ——— metadata branch ———
        self.use_metadata = bool(opt['model'].get('use_metadata', False))
        if self.use_metadata:
            # assuming metadata vector is length 8 (sex, age, 6‐hot site)
            meta_dim = opt['model'].get('meta_input_dim', 8)

            # build the 2‐layer MLP: meta_dim → 512 → feature_dim
            self.meta_mlp = nn.Sequential(
                nn.Linear(meta_dim, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(),            # Swish
                nn.Dropout(0.3),
                nn.Linear(512, feature_dim),
                nn.BatchNorm1d(feature_dim),
                nn.SiLU()
            )

            # override classifier to accept concatenated [img_feat; meta_feat]
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim * 2, opt['model']['output_neurons'])
            )

        if self.use_contrastive_head:
            print("🧩 Adding contrastive projection head.")
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 128)
            )
            self.classifier = nn.Linear(128, 1)
            #print(f"Using Supervised Contrastive Loss with temp={opt['model'].get('contrastive_temperature', 0.07)}, margin={opt['model'].get('contrastive_margin', 0.5)}")
        else:
            self.projector = None

        if self.use_svm_head or opt['model'].get('loss_function') == 'svm_hinge':
            print("🔵 Using SVM-style Linear Head instead of original classifier")
            self.svm_head = nn.Linear(feature_dim, 1)
        else:
            self.svm_head = None

        if self.freeze_backbone:
            self.freeze_layers()

    def freeze_layers(self):
        if self.is_transformer:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self._set_module_trainable(self.classifier, True)
            if self.svm_head is not None:
                self._set_module_trainable(self.svm_head, True)
            if self.projector is not None:
                self._set_module_trainable(self.projector, True)
                print("🧩 Projector head unfrozen.")
            print("Transformer backbone frozen; head remains trainable.")
        else:
            for param in self.backbone.parameters():
                param.requires_grad = False
            classifier = getattr(self.backbone, 'fc', None) or \
                         getattr(self.backbone, 'head', None) or \
                         getattr(self.backbone, 'classifier', None)
            if classifier:
                self._set_module_trainable(classifier, True)
            if self.svm_head is not None:
                self._set_module_trainable(self.svm_head, True)
            if self.projector is not None:
                self._set_module_trainable(self.projector, True)
            children = list(self.backbone.children())
            if classifier and classifier in children:
                children.remove(classifier)
            if self.num_unfrozen_layers and len(children) > 0:
                to_unfreeze = children[-self.num_unfrozen_layers:]
                for module in to_unfreeze:
                    self._set_module_trainable(module, True)
                print(f"Unfroze the last {self.num_unfrozen_layers} modules + head.")

    @staticmethod
    def _set_module_trainable(module, trainable):
        for param in module.parameters():
            param.requires_grad = trainable

    def forward(self, x, return_features=False, return_projection=False, return_logits=False):
        if isinstance(x, (list, tuple)):
            img = x[0]
            if self.use_metadata:
                meta = x[1]
        else:
            img = x

        if hasattr(self.backbone, 'forward_features'):
            features = self.backbone.forward_features(img)
        else:
            features = self.backbone(img)

        if features.ndim == 4 and features.shape[-1] != features.shape[-2]:
            features = features.permute(0, 3, 1, 2)  # [B, C, H, W]

        if features.ndim == 4:
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)
        elif features.ndim == 3:
            features = features[:, 0]
        elif features.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected feature shape: {features.shape}")
        
        if self.use_metadata:
            # meta is [B × meta_dim]
            meta_feat = self.meta_mlp(meta)    # ⇒ [B × feature_dim]
            features = torch.cat([features, meta_feat], dim=1)
            
        if return_features and return_logits:
            logits = self.classifier(features).squeeze(-1)
            return features, logits
        
        if return_features:
            return features

        if return_projection:
            if self.projector is None:
                raise ValueError("Projector requested but not defined!")
            return self.projector(features)

        if self.use_contrastive_head:
            proj = self.projector(features)
            return self.classifier(proj).squeeze(-1)

        if self.use_svm_head:
            return self.svm_head(features).squeeze(-1)

        return self.classifier(features).squeeze(-1)


def test_melanoma_model(opt, testmodel_path):
    print(f"Loading saved model: {testmodel_path}")

    model = train_melanoma_model(opt)  # Build a fresh model

    checkpoint = torch.load(testmodel_path, map_location='cpu')

    try:
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        print("⚡ Fine-tuned model detected. Adjusting model structure for loading...")

        # Rebuild model head to match fine-tuned checkpoint
        model.use_svm_head = False
        model.use_contrastive_head = False
        model.projector = None

        feature_dim = model.backbone.num_features if hasattr(model.backbone, 'num_features') else 1280
        model.classifier = nn.Sequential(
            nn.Dropout(opt['model']['dropout_rate']),
            nn.Linear(feature_dim, 1)
        )

        # Now load again
        model.load_state_dict(checkpoint)

    return model


def train_melanoma_model(opt):
    model = MelanomaModel(opt)
    return model
