import torch
import torch.nn as nn
import timm

from main import argument_parser
from model import MelanomaModel
from utils import log_model


class HybridModel(MelanomaModel):
    def __init__(self, opt):
        super(HybridModel, self).__init__(opt)

        # Hybrid model specific config
        embed_dim = opt['model']['hybrid'].get('embed_dim', 256)
        num_transformer_layers = opt['model']['hybrid'].get('num_transformer_layers', 2)
        num_heads = opt['model']['hybrid'].get('num_heads', 4)

        # General model specific config
        backbone_name = opt['model']['backbone']
        pretrained = opt['model']['pretrained']
        dropout_rate = opt['model']['dropout_rate']
        self.freeze_backbone = opt['model'].get('freeze_backbone', False)
        self.num_unfrozen_layers = opt['model'].get('num_unfrozen_layers', None)

        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)

        # Get CNN output feature size
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']

        # Project CNN output to embed_dim for the transformer
        self.projection = nn.Conv2d(self.feature_dim, embed_dim, kernel_size=1)

        # Positional encoding for flattened spatial locations
        self.pos_embedding = nn.Parameter(torch.randn(1, 49, embed_dim))  # Assuming 7x7 output

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),             # [B, embed_dim, 1]
            nn.Flatten(),                        # [B, embed_dim]
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(embed_dim // 2, 1),
        )

        if self.freeze_backbone:
            self.freeze_layers()

    def freeze_layers(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        children = list(self.backbone.children())

        if self.num_unfrozen_layers and len(children) > 0:
            to_unfreeze = children[-self.num_unfrozen_layers:]
            for module in to_unfreeze:
                self._set_module_trainable(module, True)
            print(f"Unfroze the last {self.num_unfrozen_layers} modules of the backbone (CNN)")
        else:
            print("Only transformer and classifier head remains trainable.")

    def forward(self, x):
        feats = self.backbone(x)[-1]  # [B, C, H, W]
        proj = self.projection(feats)  # [B, embed_dim, H, W]

        B, E, H, W = proj.shape
        tokens = proj.view(B, E, H * W).transpose(1, 2)  # [B, H*W, embed_dim]

        tokens = tokens + self.pos_embedding[:, :tokens.size(1), :]
        transformed = self.transformer(tokens)  # [B, H*W, embed_dim]

        transformed = transformed.transpose(1, 2)  # [B, embed_dim, H*W]
        out = self.classifier(transformed)  # [B, 1]
        return out

if __name__ == '__main__':
    # Example usage

    opt = argument_parser()

    model = HybridModel(opt)

    log_model(opt, model)

    criterion = nn.BCEWithLogitsLoss()

    # Dummy input
    x = torch.randn(32, 3, 224, 224)
    y = torch.randint(0, 2, (32, 1)).float()

    logits = model(x)
    loss = criterion(logits, y)
    print(loss)