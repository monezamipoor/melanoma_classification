# contrastive_svm.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from loss import SVMHingeLoss, SupConLoss
from utils import (
    save_checkpoint, soft_voting_probs_from_logits,
    write_kaggle_csv, log_results, log_test
)
from metrics import evaluate_metrics
from wandb_helper import wandb_watch, wandb_val_log, wandb_test_log


def contrastive_train_batch(melanomamodel, images, labels, epoch):
    melanomamodel.optimizer.zero_grad()
    images1, images2 = images
    images1, images2 = images1.to(melanomamodel.device), images2.to(melanomamodel.device)
    labels = labels.to(melanomamodel.device)

    f1 = melanomamodel.model(images1, return_features=True)
    f2 = melanomamodel.model(images2, return_features=True)

    p1 = F.normalize(melanomamodel.model.projector(f1), dim=1)
    p2 = F.normalize(melanomamodel.model.projector(f2), dim=1)

    embeddings = torch.cat([p1, p2], dim=0)
    labs = torch.cat([labels, labels], dim=0)

    temp = melanomamodel.opt['model'].get('contrastive_temperature', 0.07)
    sim = torch.matmul(embeddings, embeddings.T) / temp

    mask = (labs.unsqueeze(0) == labs.unsqueeze(1)).float()
    logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0], device=mask.device)

    exp_sim = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-9)

    mean_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-9)
    loss = -mean_pos.mean()

    loss.backward()
    melanomamodel.optimizer.step()
    return loss


def contrastive_validate(m, val_loader, epoch):
    device = m.device
    total_loss = 0.0
    in_cl = (m.opt['model']['loss_function'] == 'contrastive') and \
            (epoch < m.opt['training']['contrastive_epochs'])

    svm_fn = SVMHingeLoss().to(device)
    supcon = SupConLoss(
        temperature=m.opt['model']['contrastive_temperature'],
        margin_threshold=m.opt['model']['contrastive_margin']
    ).to(device)

    all_out, all_lbl = [], []

    with torch.no_grad():
        loop = tqdm(val_loader, desc=("[Val: Contrastive]" if in_cl else "[Val]"))
        for batch in loop:
            (x1, x2), y = batch
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            f1 = m.model(x1, return_features=True)
            f2 = m.model(x2, return_features=True)

            if m.opt['model'].get('use_contrastive_svm', False):
                o1 = m.model.svm_head(f1).squeeze()
                o2 = m.model.svm_head(f2).squeeze()
                outputs = torch.cat([o1, o2], dim=0)
                labs = torch.cat([y, y], dim=0)
                loss = svm_fn(outputs, labs)
            else:
                z1 = F.normalize(m.model.projector(f1), dim=1)
                z2 = F.normalize(m.model.projector(f2), dim=1)
                outputs = torch.cat([z1, z2], dim=0)
                labs = torch.cat([y, y], dim=0)
                loss = supcon(outputs, labs)

            total_loss += loss.item()
            all_out.append(outputs.cpu())
            all_lbl.append(labs.cpu())

    avg = total_loss / len(val_loader)

    if in_cl:
        print(f"⚡ Contrastive‐only val @ epoch {epoch}, loss={avg:.4f}")
        return avg, {}

    outs = torch.cat(all_out, dim=0)
    lbls = torch.cat(all_lbl, dim=0)
    metrics = evaluate_metrics(m.opt, outs, lbls, epoch+1)
    log_results(m.opt, metrics)
    return avg, metrics


def contrastive_validate_loss(m, total_loss, val_loader, description="[Val]"):
    device = m.device
    m.model.eval()
    all_out, all_lbl = [], []

    loop = tqdm(val_loader, desc=description)
    with torch.no_grad():
        for batch in loop:
            images, labels = batch
            labels = labels.to(device)

            if isinstance(images, (list, tuple)):
                i1, i2 = images
                i1, i2 = i1.to(device), i2.to(device)
                f1 = m.model(i1, return_features=True)
                f2 = m.model(i2, return_features=True)

                if m.opt['model'].get('use_contrastive_svm', False):
                    o1 = m.model.svm_head(f1).squeeze()
                    o2 = m.model.svm_head(f2).squeeze()
                    outputs = torch.cat([o1, o2], dim=0)
                    labs = torch.cat([labels, labels], dim=0)
                else:
                    p1 = F.normalize(m.model.projector(f1), dim=1)
                    p2 = F.normalize(m.model.projector(f2), dim=1)
                    outputs = torch.cat([p1, p2], dim=0)
                    labs = torch.cat([labels, labels], dim=0)

                loss_fn = m.criterion if m.opt['model']['loss_function'] == 'contrastive' else m.criterion_second
                loss = loss_fn(outputs, labs.float())

            else:
                imgs = images.to(device)
                outputs = m.model(imgs)
                loss = m.criterion(outputs, labels.float())
                labs = labels

            total_loss += loss.item()
            all_out.append(outputs.cpu())
            all_lbl.append(labs.cpu())

    return torch.cat(all_lbl, dim=0), torch.cat(all_out, dim=0), total_loss


def contrastive_test(opt, melanoma_model_list, val_loader, tag="natural"):
    if not melanoma_model_list:
        print("Test: No models to test. Exiting...")
        return

    predict_mode = melanoma_model_list[0].predictmode
    print(f"Test: Generate predictions only = {predict_mode}")

    outs_list, all_lbl = [], None

    for mt in melanoma_model_list:
        print(f"Test: Model {mt.model_path}")
        mt.model = mt.model.to(mt.device)
        mt.model.eval()

        total_loss = 0.0
        with torch.no_grad():
            lbls, outputs, total_loss = contrastive_validate_loss(mt, total_loss, val_loader, description='[Test]')

        if all_lbl is None:
            all_lbl = lbls

        if outputs.dim() > 1 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)
        if all_lbl.dim() > 1 and all_lbl.shape[1] == 1:
            all_lbl = all_lbl.squeeze(1)

        outs_list.append(outputs)

    logits = torch.stack(outs_list, dim=0)
    probs = soft_voting_probs_from_logits(logits)

    if predict_mode:
        print("Saving predictions only (no ground truth labels available).")
        write_kaggle_csv(opt, val_loader.dataset.files, probs, tag=tag)
    else:
        print("Evaluating test metrics...")
        metrics = evaluate_metrics(opt, probs, all_lbl, epoch="Test")
        log_test(opt, metrics, tag=tag)
        wandb_test_log(metrics, tag=tag)
        print(f"Test Metrics ({tag}): {metrics}")
