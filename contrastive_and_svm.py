import torch
import torch.nn as nn
import torch.nn.functional as F
from loss import SVMHingeLoss, SupConLoss
from tqdm import tqdm
from loss import melanoma_loss
import wandb

def train_contrastive_batch(melanomamodel, images, labels):
    images1, images2 = images
    images1, images2 = images1.to(melanomamodel.device), images2.to(melanomamodel.device)
    labels = labels.to(melanomamodel.device)

    features1 = melanomamodel.model(images1, return_features=True)
    features2 = melanomamodel.model(images2, return_features=True)

    if melanomamodel.opt['model'].get('use_contrastive_svm', False):
        outputs1 = melanomamodel.model.svm_head(features1).squeeze()
        outputs2 = melanomamodel.model.svm_head(features2).squeeze()

        outputs = torch.cat([outputs1, outputs2], dim=0)
        labels_contrastive = torch.cat([labels, labels], dim=0)

        svm_loss_fn = SVMHingeLoss()
        loss = svm_loss_fn(outputs, labels_contrastive)
    else:
        proj1 = F.normalize(melanomamodel.model.projector(features1), dim=1)
        proj2 = F.normalize(melanomamodel.model.projector(features2), dim=1)

        embeddings = torch.cat([proj1, proj2], dim=0)
        labels_contrastive = torch.cat([labels, labels], dim=0)

        temperature = melanomamodel.opt['model'].get('contrastive_temperature', 0.07)
        sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature

        label_mask = (labels_contrastive.unsqueeze(0) == labels_contrastive.unsqueeze(1)).float()
        logits_mask = torch.ones_like(label_mask) - torch.eye(label_mask.shape[0], device=label_mask.device)

        exp_sim = torch.exp(sim_matrix) * logits_mask
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

        mean_log_prob_pos = (label_mask * log_prob).sum(1) / (label_mask.sum(1) + 1e-9)
        loss = -mean_log_prob_pos.mean()

    return loss


def validate_contrastive(m, val_loader, epoch):
    device = m.device
    m.model.eval()
    total_loss = 0.0

    bce_crit = nn.BCEWithLogitsLoss()
    svm_loss_fn = SVMHingeLoss().to(device)
    supcon_crit = SupConLoss(
        temperature=m.opt['model']['contrastive_temperature'],
        margin_threshold=m.opt['model']['contrastive_margin']
    ).to(device)

    all_outputs = []
    all_labels = []

    with torch.no_grad():
        loop = tqdm(val_loader, desc="[Val: Contrastive]")
        for batch in loop:
            (x1, x2), y = batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            features1 = m.model(x1, return_features=True)
            features2 = m.model(x2, return_features=True)

            if m.opt['model'].get('use_contrastive_svm', False):
                outputs1 = m.model.svm_head(features1).squeeze()
                outputs2 = m.model.svm_head(features2).squeeze()

                outputs = torch.cat([outputs1, outputs2], dim=0)
                labels_contrastive = torch.cat([y, y], dim=0)

                loss = svm_loss_fn(outputs, labels_contrastive)
                total_loss += loss.item()

                all_outputs.append(outputs.cpu())
                all_labels.append(labels_contrastive.cpu())
            else:
                z1 = F.normalize(m.model.projector(features1), dim=1)
                z2 = F.normalize(m.model.projector(features2), dim=1)

                feats = torch.cat([z1, z2], dim=0)
                labs = torch.cat([y, y], dim=0)

                loss = supcon_crit(feats, labs)
                total_loss += loss.item()

                all_outputs.append(torch.tensor([loss.item()]))
                all_labels.append(torch.tensor([0]))  # dummy

    avg_loss = total_loss / len(val_loader)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return avg_loss, {}

def switch_to_finetune_phase(melanomamodel, epoch, fold_idx=None):
    print(f"✨ Switching to fine-tuning phase at epoch {epoch}" + (f" for Fold {fold_idx}" if fold_idx is not None else "") + "...")

    melanomamodel.opt['model']['loss_function'] = 'bce'
    melanomamodel.optimizer = melanomamodel.get_optimizer()
    melanomamodel.scheduler = melanomamodel.get_scheduler()
    # melanomamodel.criterion = melanoma_loss(melanomamodel.opt)
    melanomamodel.criterion = melanoma_loss(melanomamodel.opt).to(melanomamodel.device)


    melanomamodel.model.training_phase = 'finetune'
    melanomamodel.model.use_svm_head = False
    melanomamodel.model.use_contrastive_head = False
    melanomamodel.model.projector = None

    feature_dim = melanomamodel.model.backbone.num_features if hasattr(melanomamodel.model.backbone, 'num_features') else 1280
    melanomamodel.model.classifier = nn.Sequential(
        nn.Dropout(melanomamodel.opt['model']['dropout_rate']),
        nn.Linear(feature_dim, 1)
    )

    melanomamodel.opt['training']['contrastive_epochs'] = -1

    if melanomamodel.logwandb:
        log_dict = {"Switch_to_Finetune_Epoch": epoch}
        if fold_idx is not None:
            log_dict["Fold"] = fold_idx
        wandb.log(log_dict)

def extract_contrastive_pair(images):
    if isinstance(images, (list, tuple)) and len(images) == 2:
        return images[0]  # Return single image batch
    return images

def maybe_run_contrastive_validation(model_obj, val_loader, epoch):
    loss_fn_name = model_obj.opt['model']['loss_function']
    contrastive_epochs = model_obj.opt['training'].get('contrastive_epochs', 5)

    if loss_fn_name == 'contrastive' and epoch < contrastive_epochs:
        avg_loss, _ = validate_contrastive(model_obj, val_loader, epoch)
        print(f"⚡ Contrastive‐only val @ epoch {epoch}, loss={avg_loss:.4f}")
        return avg_loss, {}, True

    return None, None, False

def evaluate_svm_outputs(melanomamodel, images1, images2, labels):
    device = melanomamodel.device
    features1 = melanomamodel.model(images1, return_features=True)
    features2 = melanomamodel.model(images2, return_features=True)

    if melanomamodel.opt['model'].get('use_contrastive_svm', False):
        outputs1 = melanomamodel.model.svm_head(features1).squeeze()
        outputs2 = melanomamodel.model.svm_head(features2).squeeze()
        outputs = torch.cat([outputs1, outputs2], dim=0)
        labels = torch.cat([labels, labels], dim=0)
    else:
        proj1 = F.normalize(melanomamodel.model.projector(features1), dim=1)
        proj2 = F.normalize(melanomamodel.model.projector(features2), dim=1)
        outputs = torch.cat([proj1, proj2], dim=0)
        labels = torch.cat([labels, labels], dim=0)

    return outputs, labels

