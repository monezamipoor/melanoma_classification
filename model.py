"""
This module is responsible for defining model architectures for melanoma classification.
We use the timm library to create a variety of models, including transformers and CNNs.
Layers can be frozen or unfrozen based on the configuration provided in the options.
The model can be configured to use different heads for classification, including SVM-style and contrastive heads.
The model can also be configured to use a contrastive projection head for self-supervised learning.

"""


import torch
import torch.nn as nn
import timm

class MelanomaModel(nn.Module):
    # This class defines the melanoma classification model architecture.
    def __init__(self, opt):
        super(MelanomaModel, self).__init__()
        # Initialize the model with the given options.
        self.opt = opt   
        mode = opt['model'].get('mode', 'regular').lower()
        # Set the mode of the model based on the configuration.
        if mode == 'regular':
            # Regular mode: use the standard classifier head.
            self.use_svm_head = False
            self.use_contrastive_svm = False
            self.use_contrastive_head = False
        elif mode == 'svm':
            # SVM mode: use SVM-style linear head.
            self.use_svm_head = True
            self.use_contrastive_svm = False
            self.use_contrastive_head = False
        elif mode == 'contrastive':
            # Contrastive mode: use contrastive projection head.
            self.use_svm_head = False
            self.use_contrastive_svm = False
            self.use_contrastive_head = True
        elif mode == 'svm+contrastive':
            # SVM + Contrastive mode: use both SVM-style linear head and contrastive projection head.
            # This approach is deactivated by default and we arew not using it in the code anymore.
            self.use_svm_head = True
            self.use_contrastive_svm = True
            self.use_contrastive_head = True
        else:
            raise ValueError(f"Unknown mode {mode}! Must be one of: regular, svm, contrastive, svm+contrastive")

        # Set the model parameters based on the configuration.
        backbone_name = opt['model']['backbone']
        # check the pretrained model is requested
        pretrained = opt['model']['pretrained']
        # set the dropout rate
        dropout_rate = opt['model']['dropout_rate']
        # Check configuration for freezing layers
        self.freeze_backbone = opt['model'].get('freeze_backbone', False)
        # decide how many layers to unfreeze based on the configuration
        self.num_unfrozen_layers = opt['model'].get('num_unfrozen_layers', None)

        # Check if any model name contains "swin" or "vit" to determine if it's a transformer model
        self.is_transformer = any(keyword in backbone_name.lower() for keyword in ["swin", "vit"])

        # --- Backbone ---
        if self.is_transformer:
            # If the model is a transformer, we need to handle it differently.
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            # Set the number of features based on the backbone.
            feature_dim = self.backbone.num_features
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim, opt['model']['output_neurons'])
            )
            # If the model has a head, we need to replace it with our own classifier.
        else:
            # If the model is not a transformer, we can use the standard classifier head.
            # Create the backbone model using timm.
            self.backbone = timm.create_model(backbone_name, pretrained=pretrained)
            # Check the model last layer is fc classifier.
            if hasattr(self.backbone, 'fc'):
                # If the model has a fully connected layer, we need to replace it with our own classifier.
                feature_dim = self.backbone.fc.in_features
                # If the model has a global pooling layer, we need to replace it with an identity layer.
                if hasattr(self.backbone, 'global_pool'):
                    self.backbone.global_pool = nn.Identity()  
                # Replace the fully connected layer with our own classifier.
                self.backbone.fc = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(feature_dim, opt['model']['output_neurons'])
                )
                # Set the classifier to the new fully connected layer.
                self.classifier = self.backbone.fc
            # Check the model last layer is head classifier.
            # If the model has a head, we need to replace it with our own classifier.
            elif hasattr(self.backbone, 'head'):
                feature_dim = self.backbone.head.in_features
                self.backbone.head = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(feature_dim, opt['model']['output_neurons'])
                )
                # Replace the head with our own classifier.
                self.classifier = self.backbone.head
            # Check the model last layer is classifier.
            elif hasattr(self.backbone, 'classifier'):
                # If the model has a classifier, we need to replace it with our own classifier.
                feature_dim = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Sequential(
                    nn.Dropout(dropout_rate),
                    nn.Linear(feature_dim, opt['model']['output_neurons'])
                )
                self.classifier = self.backbone.classifier
            # If the model has no classifier, we need to raise an error.
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
                nn.LayerNorm(512),
                nn.SiLU(),            # Swish
                nn.Dropout(0.3),
                nn.Linear(512, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.SiLU()
            )

            # override classifier to accept concatenated [img_feat; meta_feat]
            self.classifier = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(feature_dim * 2, opt['model']['output_neurons'])
            )

        if self.use_contrastive_head:
            # If the model is using a contrastive head, we need to add a projection head.
            # The projection head is a sequential model with two linear layers and a ReLU activation.
            print(" Adding contrastive projection head.")
            self.projector = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 128)
            )
            # The classifier is a linear layer with 128 input features and 1 output feature.
            self.classifier = nn.Linear(128, 1)
        else:
            # If the model is not using a contrastive head, we need to set the projector to None.
            self.projector = None
        # If the model is using an SVM-style linear head, we need to add a linear layer with 1 output feature.
        if self.use_svm_head or opt['model'].get('loss_function') == 'svm_hinge':
            # If the model is using an SVM-style linear head, we need to add a linear layer with 1 output feature.
            print(" Using SVM-style Linear Head instead of original classifier")
            self.svm_head = nn.Linear(feature_dim, 1)
        else:
            # If the model is not using an SVM-style linear head, we need to set the SVM head to None.
            self.svm_head = None

        # If the freeze_backbone option is set to True, we need to freeze the backbone layers.
        if self.freeze_backbone:
            self.freeze_layers()

    def freeze_layers(self):
        # Freeze the backbone layers based on the configuration.
        if self.is_transformer:
            # We set the requires_grad attribute of the backbone parameters to False.
            for param in self.backbone.parameters():
                param.requires_grad = False
            # If the model has a classifier, we need to set its requires_grad attribute to True.
            self._set_module_trainable(self.classifier, True)
            if self.svm_head is not None:
                # If the model has an SVM head, we need to set its requires_grad attribute to True.
                self._set_module_trainable(self.svm_head, True)
            if self.projector is not None:
                # If the model has a projector, we need to set its requires_grad attribute to True.
                self._set_module_trainable(self.projector, True)
                print(" Projector head unfrozen.")
            print("Transformer backbone frozen; head remains trainable.")
        else:
            # We set the requires_grad attribute of the backbone parameters to False.
            for param in self.backbone.parameters():
                param.requires_grad = False
            # If the model has a classifier, we need to set its requires_grad attribute to True.
            classifier = getattr(self.backbone, 'fc', None) or \
                         getattr(self.backbone, 'head', None) or \
                         getattr(self.backbone, 'classifier', None)
            if classifier:
                # If the model has a classifier, we need to set its requires_grad attribute to True.
                self._set_module_trainable(classifier, True)
            if self.svm_head is not None:
                # If the model has an SVM head, we need to set its requires_grad attribute to True.
                self._set_module_trainable(self.svm_head, True)
            if self.projector is not None:
                # If the model has a projector, we need to set its requires_grad attribute to True.
                self._set_module_trainable(self.projector, True)
            # We extract the list of children modules from the backbone.
            # If the model has a classifier, we need to remove it from the list of children.    
            children = list(self.backbone.children())
            if classifier and classifier in children:
                children.remove(classifier)
            #We check if the number of unfrozen layers is set in the configuration.
            # If the number of unfrozen layers is set, we need to unfreeze the last layers.    
            if self.num_unfrozen_layers and len(children) > 0:
                # We unfreeze the last num_unfrozen_layers layers of the backbone.
                # We set the requires_grad attribute of the last num_unfrozen_layers layers to True.
                to_unfreeze = children[-self.num_unfrozen_layers:]
                for module in to_unfreeze:
                    self._set_module_trainable(module, True)
                print(f"Unfroze the last {self.num_unfrozen_layers} modules + head.")

    @staticmethod
    def _set_module_trainable(module, trainable):
        # Set the requires_grad attribute of the module parameters to the given value.
        for param in module.parameters():
            # If the module is a Sequential model, we need to set the requires_grad attribute of its parameters.
            param.requires_grad = trainable

    def forward(self, x, return_features=False, return_projection=False, return_logits=False):
        # Forward pass through the model.
        # The input x can be a batch of images or a list of images.
        if isinstance(x, (list, tuple)):
            img = x[0]
            if self.use_metadata:
                meta = x[1]
        else:
            img = x
        if hasattr(self.backbone, 'forward_features'):
            # If the model has a forward_features method, we need to use it to extract the features.
            features = self.backbone.forward_features(img)
        else:
            # If the model does not have a forward_features method, we need to use the standard forward method.
            features = self.backbone(img)
        # If the model has a global pooling layer, we need to apply it to the features.
        if features.ndim == 4 and features.shape[-1] != features.shape[-2]:
            # if the model is swin transformer, we need to permute the features.
            features = features.permute(0, 3, 1, 2)  

        if features.ndim == 4:
            # If the model has a 4D feature map, we need to apply global average pooling.
            features = torch.nn.functional.adaptive_avg_pool2d(features, 1)
            features = features.view(features.size(0), -1)
        elif features.ndim == 3:
            # If the model has a 3D feature map, we need to apply global average pooling.
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
            # If the return_features option is set to True, we need to return the features.
            # We use it for contrastive learning.
            return features

        if return_projection:
            if self.projector is None:
                # If the projector is not defined, we need to raise an error.
                raise ValueError("Projector requested but not defined!")
            return self.projector(features)

        if self.use_contrastive_head:
            # If the model is using a contrastive head, we need to apply the projector to the features.
            proj = self.projector(features)
            return self.classifier(proj).squeeze(-1)

        if self.use_svm_head:
            # If the model is using an SVM-style linear head, we need to apply it to the features.
            return self.svm_head(features).squeeze(-1)

        return self.classifier(features).squeeze(-1)


def test_melanoma_model(opt, testmodel_path):
    # Load the best model from the specified path.
    print(f"Loading saved model: {testmodel_path}")
    # Create the model using the provided options.
    model = train_melanoma_model(opt) 
    # Load the model state dictionary from the specified path.
    checkpoint = torch.load(testmodel_path, map_location='cpu')

    try:
        # Attempt to load the state dictionary into the model.
        model.load_state_dict(checkpoint)
    except RuntimeError as e:
        # If there is a mismatch in the model structure, we need to adjust the model.
        print(" Fine-tuned model detected. Adjusting model structure for loading...")

        # Rebuild model head to match fine-tuned checkpoint
        # Remove the classifier and projector from the model.
        model.use_svm_head = False
        # Remove the SVM head from the model.
        model.use_contrastive_head = False
        # Remove the contrastive head from the model.
        model.projector = None

        feature_dim = model.backbone.num_features if hasattr(model.backbone, 'num_features') else 1280
        # Set the classifier to a new sequential model with a dropout layer and a linear layer.
        model.classifier = nn.Sequential(
            nn.Dropout(opt['model']['dropout_rate']),
            nn.Linear(feature_dim, 1)
        )

        # Now load again
        model.load_state_dict(checkpoint)

    return model


def train_melanoma_model(opt):
    # Create the model using the provided options.
    model = MelanomaModel(opt)
    return model
