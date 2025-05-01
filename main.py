import argparse
import yaml
import os
import torch
import torch.cuda.amp as amp
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import random
import numpy as np
from data import melanoma_train_dataloaders, melanoma_test_dataloaders
from debug import print_batch_label_dist, print_raw_logits_and_probs
from model import train_melanoma_model, test_melanoma_model
from model_hybrid import train_hybrid_model, test_hybrid_model
from loss import melanoma_loss, FocalLoss, DiceLoss
from utils import (
    log_results, cuda_available, log_model, save_checkpoint,
    write_kaggle_csv, soft_voting_probs_from_logits, log_test,
    save_augmented_samples
)
from metrics import evaluate_metrics
from wandb_helper import (
    wandb_login, wandb_watch, wandb_train_log,
    wandb_val_log, wandb_test_log
)
import math
import wandb
from copy import deepcopy

# Contrastive logic moved to contrastive_svm module
from contrastive_svm import ContrastiveSVM

#Uncomment to turn off wandb entirely for debugging only
#wandb.init(mode="disabled")


class MelanomaTest:
    def __init__(self, opt, testmodel):
        self.opt = opt
        print(opt)

        self.device = cuda_available(self.opt)

        self.predictmode, self.val_loader = melanoma_test_dataloaders(opt)
        self.is_kfold = False

        if opt['model']['hybrid'].get('enabled', False):           
            self.model = test_hybrid_model(opt, testmodel).to(self.device)
            print("Using Hybrid Model")
        else:                                                      
            self.model = test_melanoma_model(opt, testmodel).to(self.device)

        self.model_path = testmodel
        self.criterion = melanoma_loss(opt).to(self.device)

        self.best_metrics = {metric: float('-inf') for metric in opt['testing']['model_save_metrics']}

        self.logwandb = wandb_login(opt)
        print("Wandb logging: ", self.logwandb)

        log_model(self.opt, self.model)

class MelanomaTrainer:
    def __init__(self, opt):
        self.opt = opt
        print(opt)
        self.device = cuda_available(self.opt)
        self.cengine = ContrastiveSVM(opt, self.device)
        # K-Fold 
        if opt['dataset'].get('use_groupkfold', False):
            self.fold_loaders = melanoma_train_dataloaders(opt)
            self.is_kfold = True
        else:
            self.train_loader, self.val_loader, self.val_loader_balanced = melanoma_train_dataloaders(opt)
            self.is_kfold = False

        if opt['model']['hybrid'].get('enabled', False):
            print("Using Hybrid Model")
            self.model = train_hybrid_model(opt).to(self.device)
        else:
            self.model = train_melanoma_model(opt).to(self.device)

        loss_fn = opt['model'].get('loss_function', 'bce').lower()
        if loss_fn == 'contrastive':
            # 1) supervised‐contrastive (or hinge) during the pre‐training phase
            self.criterion = melanoma_loss(opt, self.train_loader).to(self.device)

            # 2) plain BCE for the fine‐tune / classifier head
            bce_opt = deepcopy(opt)
            bce_opt['model']['loss_function'] = 'bce'
            self.criterion_second = melanoma_loss(bce_opt, self.train_loader).to(self.device)
        else:
            # single‐loss-mode
            self.criterion = melanoma_loss(opt, self.train_loader).to(self.device)

        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.scaler = amp.GradScaler() if opt['training']['mixed_precision'] else None
        self.best_metrics = self.reset_metrics()

        self.freeze_backbone_epochs = opt['training'].get('freeze_backbone_epochs', 0)
        self.backbone_frozen = self.freeze_backbone_epochs > 0

        if self.backbone_frozen and not (self.opt['model'].get('loss_function') == 'contrastive'):
            print(f"🔒 Freezing backbone for {self.freeze_backbone_epochs} epochs...")
            self.freeze_backbone(True)

        self.logwandb = wandb_login(opt)
        print("Wandb: ", self.logwandb)

        log_model(self.opt, self.model)

    def reset_metrics(self):
        return {metric: float('-inf') for metric in self.opt['testing']['model_save_metrics']}

    def get_optimizer(self):
        if self.opt['training']['optimizer'] == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.opt['training']['learning_rate'])
        elif self.opt['training']['optimizer'] == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.opt['training']['learning_rate'], momentum=0.9)
        elif self.opt['training']['optimizer'] == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.opt['training']['learning_rate'])
        elif self.opt['training']['optimizer'] == 'adagrad':
            return optim.Adagrad(self.model.parameters(), lr=self.opt['training']['learning_rate'])
        elif self.opt['training']['optimizer'] == 'amsgrad':
            return optim.Adam(self.model.parameters(), lr=self.opt['training']['learning_rate'], amsgrad=True)

    def get_scheduler(self):
        if self.opt['training']['scheduler'] == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt['training']['epochs'])
        elif self.opt['training']['scheduler'] == 'step':
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[self.opt['training']['step_size']],
                gamma=self.opt['training']['decay_rate']
            )
        elif self.opt['training']['scheduler'] == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1, verbose=True)
        else:
            return None

    def freeze_backbone(self, freeze=False):
        for param in list(self.model.parameters())[:-1]:
            param.requires_grad = not freeze
        print("Backbone layers frozen?= " + str(freeze))


def train(melanomamodel):
    print("Starting Training")
    wandb_watch(melanomamodel.model, melanomamodel.criterion, log_freq=10)
    testmodels = []

    if melanomamodel.is_kfold:
        for fold_data in melanomamodel.fold_loaders:
            fold_idx = fold_data['fold']
            print(f"\n[INFO] Starting Fold {fold_idx}")

            melanomamodel.model = train_melanoma_model(melanomamodel.opt).to(melanomamodel.device)
            melanomamodel.best_metrics = melanomamodel.reset_metrics()

            train_loader = fold_data['train_loader']
            val_loader = fold_data['val_loader']
            val_loader_balanced = fold_data['val_loader_balanced']

            wandb_watch(melanomamodel.model, melanomamodel.criterion, log_freq=10)
            modeltokeep = None

            for epoch in range(melanomamodel.opt['training']['epochs']):
                melanomamodel.model.train()
                total_loss = 0.0

                loop = tqdm(train_loader, desc=f"[Fold {fold_idx}] Epoch {epoch+1}/{melanomamodel.opt['training']['epochs']}")
                for images, labels in loop:
                    loss = train_batch(melanomamodel, images, labels, epoch)
                    total_loss += loss.item()
                    loop.set_postfix(loss=loss.item())

                avg_loss = total_loss / len(train_loader)
                val_loss, val_metrics = validate(melanomamodel, val_loader, epoch)
                val_loss_bal, val_metrics_bal = validate(melanomamodel, val_loader_balanced, epoch)

                if melanomamodel.scheduler:
                    if isinstance(melanomamodel.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        melanomamodel.scheduler.step(val_loss)
                    else:
                        melanomamodel.scheduler.step()

                print(f"[Fold {fold_idx}] Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
                print(f"    [Natural] Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")
                print(f"    [Balanced] Val Loss: {val_loss_bal:.4f}, Balanced Metrics: {val_metrics_bal}")

                wandb_val_log(avg_loss, val_loss, val_loss_bal, val_metrics, val_metrics_bal)

                # strip contrastive components if needed (unchanged)
                if melanomamodel.opt['model']['loss_function'] == 'contrastive' and \
                   epoch + 1 == melanomamodel.opt['training']['contrastive_epochs']:
                    print("🧪 Removing contrastive components before saving...")
                    melanomamodel.model.projector = None
                    melanomamodel.model.use_contrastive_head = False
                    melanomamodel.model.use_svm_head = False
                    melanomamodel.model.training_phase = 'finetune'
                    feature_dim = (melanomamodel.model.backbone.num_features
                                  if hasattr(melanomamodel.model.backbone, 'num_features')
                                  else 1280)
                    melanomamodel.model.classifier = nn.Sequential(
                        nn.Dropout(melanomamodel.opt['model']['dropout_rate']),
                        nn.Linear(feature_dim, 1)
                    ).to(melanomamodel.device)
                    melanomamodel.criterion = melanomamodel.criterion_second
                    melanomamodel.opt['model']['loss_function'] = 'bce'

                checkpointmodel = save_checkpoint(
                    melanomamodel.opt,
                    melanomamodel.best_metrics,
                    melanomamodel.model,
                    epoch + 1,
                    val_metrics,
                    fold_idx
                )
                if checkpointmodel:
                    modeltokeep = checkpointmodel

            if modeltokeep:
                testmodels.append(modeltokeep)

    else:
        for epoch in range(melanomamodel.opt['training']['epochs']):
            melanomamodel.model.train()
            total_loss = 0

            loop = tqdm(melanomamodel.train_loader, desc=f"Epoch {epoch+1}/{melanomamodel.opt['training']['epochs']}")
            for images, labels in loop:
                loss = train_batch(melanomamodel, images, labels, epoch)
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(melanomamodel.train_loader)
            val_loss, val_metrics = validate(melanomamodel, melanomamodel.val_loader, epoch)
            val_loss_bal, val_metrics_bal = validate(melanomamodel, melanomamodel.val_loader_balanced, epoch)

            if melanomamodel.scheduler:
                if isinstance(melanomamodel.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    melanomamodel.scheduler.step(val_loss)
                else:
                    melanomamodel.scheduler.step()

            print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
            print(f"    [Natural] Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")
            print(f"    [Balanced] Val Loss: {val_loss_bal:.4f}, Metrics: {val_metrics_bal}")

            wandb_val_log(avg_loss, val_loss, val_loss_bal, val_metrics, val_metrics_bal)

            if melanomamodel.opt['model']['loss_function'] == 'contrastive' and \
               epoch + 1 == melanomamodel.opt['training']['contrastive_epochs']:
                print("🧪 Removing contrastive components before saving...")
                melanomamodel.model.projector = None
                melanomamodel.model.use_contrastive_head = False
                melanomamodel.model.use_svm_head = False
                melanomamodel.model.training_phase = 'finetune'
                feature_dim = (melanomamodel.model.backbone.num_features
                              if hasattr(melanomamodel.model.backbone, 'num_features')
                              else 1280)
                melanomamodel.model.classifier = nn.Sequential(
                    nn.Dropout(melanomamodel.opt['model']['dropout_rate']),
                    nn.Linear(feature_dim, 1)
                ).to(melanomamodel.device)
                melanomamodel.criterion = melanomamodel.criterion_second
                melanomamodel.opt['model']['loss_function'] = 'bce'

            savedmodel = save_checkpoint(
                melanomamodel.opt,
                melanomamodel.best_metrics,
                melanomamodel.model,
                epoch + 1,
                val_metrics
            )
            if savedmodel:
                testmodels = [savedmodel]

    return testmodels


def train_batch(melanomamodel, images, labels, epoch):
    lf = melanomamodel.opt['model'].get('loss_function','bce').lower()
    if lf == 'contrastive' and epoch < melanomamodel.opt['training']['contrastive_epochs']:
        return melanomamodel.cengine.train_batch(melanomamodel, images, labels, epoch)

    melanomamodel.optimizer.zero_grad()
    if isinstance(images, (list, tuple)):
        images = images[0]
    images = images.to(melanomamodel.device)
    labels = labels.to(melanomamodel.device)

    preds = melanomamodel.model(images)
    loss = melanomamodel.criterion(preds, labels.float())
    loss.backward()
    melanomamodel.optimizer.step()

    return loss


def validate(m, val_loader, epoch=1):
    """
    Wrapper that routes to contrastive or standard validation.
    """
    lf = m.opt['model']['loss_function']
    if lf == 'contrastive' and epoch < m.opt['training']['contrastive_epochs']:
        return m.cengine.validate(m, val_loader, epoch)

    device = m.device
    m.model.eval()
    total_loss = 0.0
    bce_crit = nn.BCEWithLogitsLoss()
    all_outputs, all_labels = [], []

    with torch.no_grad():
        loop = tqdm(val_loader, desc="[Val]")
        for images, labels in loop:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images, labels = images.to(device), labels.to(device)
            preds = m.model(images)
            probs = torch.sigmoid(preds)
            loss = bce_crit(preds, labels.float())
            total_loss += loss.item()
            all_outputs.append(probs.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(val_loader)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    metrics = evaluate_metrics(m.opt, all_outputs, all_labels, epoch+1)
    log_results(m.opt, metrics)

    return avg_loss, metrics


def validate_loss(melanomamodel, total_loss, val_loader, description="[Val]"):
    """
    Wrapper for contrastive or standard validate_loss.
    """
    if melanomamodel.opt['model']['loss_function'] == 'contrastive':
        return melanomamodel.cengine.validate_loss(melanomamodel, total_loss, val_loader, description)

    device = melanomamodel.device
    melanomamodel.model.eval()
    all_outputs, all_labels = [], []
    loop = tqdm(val_loader, desc=description)

    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = melanomamodel.model(images)
            loss = melanomamodel.criterion(outputs, labels.float())
            total_loss += loss.item()
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_labels, all_outputs, total_loss


def test(opt, melanoma_model_list, val_loader, tag="natural"):
    """
    Wrapper to route to contrastive or standard test.
    """
    if opt['model']['loss_function'] == 'contrastive':
        return melanoma_model_list[0].trainer.cengine.test(
           opt, melanoma_model_list, val_loader, tag)

    if melanoma_model_list is None or len(melanoma_model_list) == 0:
        print("Test: No models to test. Exiting...")
        return

    predictonly = melanoma_model_list[0].predictmode
    print(f"Test: Generate predictions only = {predictonly}")

    output_list = []
    all_labels = None

    for melanoma_test in melanoma_model_list:
        print(f"Test: Model {melanoma_test.model_path}")
        melanoma_test.model = melanoma_test.model.to(melanoma_test.device)
        melanoma_test.model.eval()

        total_loss = 0.0
        with torch.no_grad():
            labels, outputs, total_loss = validate_loss(melanoma_test, total_loss, val_loader, description='[Test]')

        if all_labels is None:
            all_labels = labels

        if outputs.dim() > 1 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)
        if all_labels.dim() > 1 and all_labels.shape[1] == 1:
            all_labels = all_labels.squeeze(1)

        output_list.append(outputs)

    ensemble_logits = torch.stack(output_list, dim=0)
    probabilities = soft_voting_probs_from_logits(ensemble_logits)

    if predictonly:
        print("Saving predictions only (no ground truth labels available).")
        write_kaggle_csv(opt, val_loader.dataset.files, probabilities, tag=tag)
    else:
        print("Evaluating test metrics...")
        metrics = evaluate_metrics(opt, probabilities, all_labels, epoch="Test")
        log_test(opt, metrics, tag=tag)
        wandb_test_log(metrics, tag=tag)
        print(f"Test Metrics ({tag}): {metrics}")


def argument_parser():
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

    return opt

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Tradeoff: slows down but ensures deterministic behavior

def main():
    set_seed(42)
    opt = argument_parser()
    testmodels = opt['dataset']['savedmodel']   # testmodels is a list of saved model paths. Multiple models (e.g. k-fold) will trigger prediction voting in test

    # Check to see if we should train first
    if testmodels is None or len(testmodels) == 0:           # Train because we don't have a model to test against
        print("TRAIN MODEL MODE")
        melanomamodel = MelanomaTrainer(opt)
        testmodels = train(melanomamodel)

        if testmodels is None or len(testmodels) == 0:
            print("Training complete. No Saved Model. Exiting.")
            return          # Nothing to test

    # 🔄 Ensure contrastive mode is turned off during testing
    if opt['model']['loss_function'] == 'contrastive':
        print("🧪 Switching loss_function from 'contrastive' to 'bce' for test phase.")
        opt['model']['loss_function'] = 'bce'
        opt['model']['mode'] = 'regular'  # Optional: depends on how your dataloader uses this

    # Test Loop begins
    melanomatests = []
    for model in testmodels:
        melanomatests.append(MelanomaTest(opt, model))      # TODO Optimise the MelanomaTest creation to be at the point of first use in test cycle

    print("=== Natural test ===")
    test(opt, melanomatests, melanomatests[0].val_loader, tag="natural")

    if opt['dataset'].get('dataset_balanced_test_csv'):
        print("\n=== Balanced test ===")
        opt['dataset']['dataset_test_csv'] = opt['dataset']['dataset_balanced_test_csv']
        balanced_tests = [MelanomaTest(opt, mt.model_path) for mt in melanomatests]
        test(opt, balanced_tests, balanced_tests[0].val_loader, tag="balanced")


if __name__ == "__main__":
    main()
