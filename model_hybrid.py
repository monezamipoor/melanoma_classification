import argparse
import os

import torch
import torch.nn as nn
import timm
import yaml

from model import MelanomaModel
from utils import log_model

# CNN as a feature extractor >> Transformer with MH attention >> Binary Classifier
class HybridModel(MelanomaModel):
    def __init__(self, opt):
        super(HybridModel, self).__init__(opt)

        # Hybrid model specific config
        cfg_embed_dim = opt['model']['hybrid'].get('embed_dim', 256)
        cfg_num_transformer_layers = opt['model']['hybrid'].get('num_transformer_layers', 2)
        cfg_num_heads = opt['model']['hybrid'].get('num_heads', 4)

        print(f"Transformer config: Embedded Dims={cfg_embed_dim}, Layers={cfg_num_transformer_layers}, Heads={cfg_num_heads}")

        # General model config
        backbone_name = opt['model']['backbone']            # Uses same param as CNN only variant to specify the CNN backbone
        pretrained = opt['model']['pretrained']
        dropout_rate = opt['model']['dropout_rate']
        self.freeze_backbone = opt['model'].get('freeze_backbone', False)
        self.num_unfrozen_layers = opt['model'].get('num_unfrozen_layers', None)

        # Creating a backbone that outputs features only to feed into the transformer section of the model.
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)

        # Fetch the last feature map from the CNN and store the number of channels for the upcoming embedding step.
        self.feature_dim = self.backbone.feature_info[-1]['num_chs']

        # Convert the output feature map to the configutred embedding dimension (linear via kernel)
        self.projection = nn.Conv2d(self.feature_dim, cfg_embed_dim, kernel_size=1)

        # Trainable CLS
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg_embed_dim))

        # Positional encoding for 50 tokens (1 CLS + 49 spatial tokens). 1 batch x spatial 7*7 + CLS
        self.pos_embedding = nn.Parameter(torch.randn(1, 50, cfg_embed_dim))

        # Create the transformer layers, takes the YML configured heads and layers as params
        encoder_layer = nn.TransformerEncoderLayer(d_model=cfg_embed_dim, nhead=cfg_num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg_num_transformer_layers)

        # Classification
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(cfg_embed_dim, cfg_embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(cfg_embed_dim // 2, 1),       # No sigmoid as that is dependent on the loss function
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

    def forward(self, batchin):
        feats = self.backbone(batchin)[-1]  # Run the CNN head to produce feature maps from final layer - SHAPE = [B, C, H, W]
        proj = self.projection(feats)  # Run the linear conversion of features to embedding by batch - SHAPE = [B, embed_dim, H, W]

        B, E, H, W = proj.shape
        tokens = proj.view(B, E, H * W).transpose(1, 2)  # Reshape the embedded structure to be token based - SHAPE = [B, H*W, embed_dim]

        cls_token = self.cls_token.expand(B, -1, -1)  # Create the CLS token to accompany the feature tokens - SHAPE = [B, 1, embed_dim]
        tokens = torch.cat((cls_token, tokens), dim=1)  # Add the CLS into the tensor - SHAPE = [B, 1+H*W, embed_dim]
        tokens = tokens + self.pos_embedding[:, :tokens.size(1), :] # Add positional encoding to the feature tokens themselves

        transformed = self.transformer(tokens)  # Actual forward pass throught the transformer blocks

        batchin = transformed[:, 0]       # Fetch the CLS token output - SHAPE = [B, embed_dim]

        logits = self.classifier(batchin)  # Classifier pass

        out = logits.squeeze(1)  # Remove redundant dimension
        return out


def test_hybrid_model(opt, testmodel):

    model = HybridModel(opt)

    if testmodel is not None:
        print('Loading saved model: ', testmodel)
        model.load_state_dict(torch.load(testmodel))

    return model

def train_hybrid_model(opt):
    model = HybridModel(opt)
    return model

# TEST HARNESS ONLY
if __name__ == '__main__':

    # Stand-alone arguements parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt", type=str, default="default.yml", help="the option file")
    parser.add_argument("-s", "--savedmodel", type=str, required=False, help="the model file to test", nargs='+')
    parser.add_argument("-t", "--testcsv", type=str, required=False, help="the csv file to test")
    args = parser.parse_args()

    if not os.path.isabs(args.opt) and not args.opt.startswith('./'):
        args.opt = os.path.join("./options", args.opt)
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    opt['opt'] = args.opt

    if args.savedmodel:
        opt['dataset']['savedmodel'] = args.savedmodel
    else:
        opt['dataset']['savedmodel'] = None
    if args.testcsv:
        opt['dataset']['dataset_test_csv'] = args.testcsv

    # args parse end

    model = HybridModel(opt)

    log_model(opt, model)

    criterion = nn.BCEWithLogitsLoss()

    # Dummy input
    x = torch.randn(32, 3, 224, 224)
    y = torch.randint(0, 2, (32, 1)).float()

    logits = model(x)
    loss = criterion(logits, y.squeeze(1))

    print(loss)