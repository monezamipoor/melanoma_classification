import argparse
import yaml
import os
import torch
import torch.cuda.amp as amp
from torch import optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from data import melanoma_train_dataloaders, melanoma_test_dataloaders
from debug import print_batch_label_dist, print_raw_logits_and_probs
from model import train_melanoma_model, test_melanoma_model
from model_hybrid import train_hybrid_model, test_hybrid_model
from loss import melanoma_loss, SVMHingeLoss, FocalLoss, DiceLoss, SupConLoss
from utils import log_results, cuda_available, log_model, save_checkpoint, write_kaggle_csv, \
    soft_voting_probs_from_logits, log_test, save_augmented_samples
from metrics import evaluate_metrics
from wandb_helper import wandb_login, wandb_watch, wandb_train_log, wandb_val_log, wandb_test_log
import math
import wandb
from contrastive_and_svm import (
  train_contrastive_batch, validate_contrastive, switch_to_finetune_phase,
  extract_contrastive_pair, maybe_run_contrastive_validation,
  evaluate_svm_outputs
)
#Uncomment to turn off wandb entirely for debugging only
#wandb.init(mode="disabled")



class MelanomaTest:
    def __init__(self, opt, testmodel):
        self.opt = opt
        print(opt)

        self.device = cuda_available(self.opt)

        # Load test dataset (natural or balanced, depends on opt setting)
        self.predictmode, self.val_loader = melanoma_test_dataloaders(opt)
        self.is_kfold = False

        if opt['model']['hybrid'].get('enabled', False):            # Use the hybrid model config
            self.model = test_hybrid_model(opt, testmodel).to(self.device)
            print("Using Hybrid Model")
        else:                                                       # Use the basic model config
            self.model = test_melanoma_model(opt, testmodel).to(self.device)

        self.model_path = testmodel
        self.criterion = melanoma_loss(opt).to(self.device)

        # Metrics tracking
        self.best_metrics = {metric: float('-inf') for metric in opt['testing']['model_save_metrics']}

        # W&B logging
        self.logwandb = wandb_login(opt)
        print("Wandb logging: ", self.logwandb)

        # Log model architecture (useful in WandB or console)
        log_model(self.opt, self.model)



class MelanomaTrainer:
    def __init__(self, opt):
        self.opt = opt
        print(opt)
        self.device = cuda_available(self.opt)

        # K-Fold 
        if opt['dataset'].get('use_groupkfold', False):

            self.fold_loaders = melanoma_train_dataloaders(opt)  # e.g. [{'fold': 0, 'train_loader': ..., 'val_loader': ...}, ...]
            self.is_kfold = True
        else:
            self.train_loader, self.val_loader, self.val_loader_balanced = melanoma_train_dataloaders(opt)
            #print_batch_label_dist(self.train_loader)
            self.is_kfold = False

        if opt['model']['hybrid'].get('enabled', False):            # Use the hybrid model config
            print("Using Hybrid Model")
            self.model = train_hybrid_model(opt).to(self.device)
        else:                                                       # Use the basic model config
            self.model = train_melanoma_model(opt).to(self.device)

        self.criterion = melanoma_loss(opt, self.train_loader).to(self.device)
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.scaler = amp.GradScaler() if opt['training']['mixed_precision'] else None
        self.best_metrics = self.reset_metrics()

        self.freeze_backbone_epochs = opt['training'].get('freeze_backbone_epochs', 0)
        self.backbone_frozen = self.freeze_backbone_epochs > 0

        # Postpone freezing if using contrastive learning
        if self.backbone_frozen and not (self.opt['model'].get('loss_function') == 'contrastive'):
            print(f"🔒 Freezing backbone for {self.freeze_backbone_epochs} epochs...")
            self.freeze_backbone(True)


        '''if opt['training']['freeze_pretrained']:
            self.freeze_backbone(bool(opt['training']['freeze_pretrained']))
        else:
            self.freeze_backbone(False)'''



        self.logwandb = wandb_login(opt)  # Track if we have an active wandb login
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

            return torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[self.opt['training']['step_size']], gamma=self.opt['training']['decay_rate'])
        elif self.opt['training']['scheduler'] == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1, verbose=True)
        else:
            return None

    def freeze_backbone(self, freeze=False):
        for param in list(self.model.parameters())[:-1]:
            param.requires_grad = not freeze
        print("Backbone layers frozen?= " + str(freeze))

def train(melanomamodel):
    #summary(self.model, input_size=(1, 3, 224, 224))       # Quick print of model arch if needed
    print("Starting Training")
    wandb_watch(melanomamodel.model, melanomamodel.criterion, log_freq=10)

    '''
    #debug layer freezing
    for name, param in melanomamodel.model.named_parameters():
        if not param.requires_grad:
            print(f"Layer {name} is frozen.")
        else:
            print(f"Layer {name} is trainable.")
    # end debug
    '''

    #We are going to return the paths to our best models for the test loop
    testmodels = []

    if melanomamodel.is_kfold:
        testmodels = []
    
        for fold_data in melanomamodel.fold_loaders:
            fold_idx = fold_data['fold']
            print(f"\n[INFO] Starting Fold {fold_idx}")

            # Re-initialize the model for each fold (bassically a fresh start):
            melanomamodel.model = train_melanoma_model(melanomamodel.opt).to(melanomamodel.device)
            melanomamodel.optimizer = melanomamodel.get_optimizer()
            melanomamodel.scheduler = melanomamodel.get_scheduler()
            melanomamodel.best_metrics = melanomamodel.reset_metrics()      # Fixed for folds using past folds metrics for comparison.
            melanomamodel.criterion = melanoma_loss(melanomamodel.opt)
            melanomamodel.best_metrics = {metric: float('-inf') for metric in melanomamodel.opt['testing']['model_save_metrics']}
    
            train_loader = fold_data['train_loader']
            val_loader = fold_data['val_loader']
            val_loader_balanced   = fold_data['val_loader_balanced']
    
            wandb_watch(melanomamodel.model, melanomamodel.criterion, log_freq=10)

            modeltokeep = None

            for epoch in range(melanomamodel.opt['training']['epochs']):
                if (melanomamodel.opt['model'].get('loss_function') == 'contrastive' and 
                    epoch == melanomamodel.opt['training'].get('contrastive_epochs', 5)):
                    switch_to_finetune_phase(melanomamodel, epoch, fold_idx if melanomamodel.is_kfold else None)

                    
                melanomamodel.model.train()
                total_loss = 0.0
    
                loop = tqdm(train_loader, desc=f"[Fold {fold_idx}] Epoch {epoch+1}/{melanomamodel.opt['training']['epochs']}")
    
                aug = melanomamodel.opt['dataset'].get('augmentations', {})
                if aug.get('save_augmentation', False):
                    save_augmented_samples(
                        train_loader,
                        num_samples=aug.get('sample_number', 10),
                        ncols=10,
                        save_dir=aug.get('aug_save_dir', '/tmp')
                    )
    
                for images, labels in loop:
                    loss = train_batch(melanomamodel, images, labels, epoch)
    
                    if melanomamodel.opt['training']['gradient_clipping']:
                        torch.nn.utils.clip_grad_norm_(melanomamodel.model.parameters(), melanomamodel.opt['training']['gradient_clipping'])
    
                    total_loss += loss.item()
                    loop.set_postfix(loss=loss.item())
    
                wandb_train_log(epoch+1, float(loss))
    
                avg_loss = total_loss / len(train_loader)
    
                val_loss, val_metrics = validate(melanomamodel, val_loader, epoch)
                val_loss_bal, val_metrics_bal = validate(melanomamodel, val_loader_balanced, epoch)

                # Step the scheduler if applicable
                if melanomamodel.scheduler is not None:
                    if isinstance(melanomamodel.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        melanomamodel.scheduler.step(val_loss)
                    else:
                        melanomamodel.scheduler.step()

                print(f"[Fold {fold_idx}] Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
                print(f"    [Natural] Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")
                print(f"    [Balanced] Val Loss: {val_loss_bal:.4f}, Balanced Metrics: {val_metrics_bal}")

                # Log validation results to wandb
                wandb_val_log(avg_loss, val_loss, val_loss_bal, val_metrics, val_metrics_bal)

                # Save checkpoint for best model or last, etc.
                checkpointmodel = save_checkpoint(melanomamodel.opt, melanomamodel.best_metrics, melanomamodel.model, epoch + 1, val_metrics, fold_idx)
                if checkpointmodel is not None:
                    modeltokeep = checkpointmodel

            # Save the model for test for this fold
            if modeltokeep is not None:
                testmodels.append(modeltokeep)


    else:
      # Single train/val scenario
      print("Starting Training")
      wandb_watch(melanomamodel.model, melanomamodel.criterion, log_freq=10)

      for epoch in range(melanomamodel.opt['training']['epochs']):
            if melanomamodel.backbone_frozen and epoch == melanomamodel.freeze_backbone_epochs:
              print(f"🔓 Unfreezing backbone at epoch {epoch}...")
              melanomamodel.freeze_backbone(False)
              melanomamodel.backbone_frozen = False


            if (melanomamodel.opt['model'].get('loss_function') == 'contrastive' and 
                epoch == melanomamodel.opt['training'].get('contrastive_epochs', 5)):
                switch_to_finetune_phase(melanomamodel, epoch, fold_idx if melanomamodel.is_kfold else None)


            melanomamodel.model = melanomamodel.model.to(melanomamodel.device)
            melanomamodel.model.train()
            total_loss = 0

            loop = tqdm(melanomamodel.train_loader, desc=f"Epoch {epoch + 1}/{melanomamodel.opt['training']['epochs']}")
            #If you want to see the images after Aug
            aug = melanomamodel.opt['dataset'].get('augmentations', {})
            if aug.get('save_augmentation', False):
                save_augmented_samples(
                    melanomamodel.train_loader,
                    num_samples=aug.get('sample_number', 10),    # default to 10 if missing
                    ncols=10,
                    save_dir=aug.get('aug_save_dir', '/tmp')
                )



            for images, labels in loop:
                loss = train_batch(melanomamodel, images, labels, epoch)

                if melanomamodel.opt['training']['gradient_clipping']:
                    torch.nn.utils.clip_grad_norm_(melanomamodel.model.parameters(), melanomamodel.opt['training']['gradient_clipping'])

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())


            wandb_train_log(epoch+1, float(loss))

            avg_loss = total_loss / len(melanomamodel.train_loader)
            val_loss, val_metrics = validate(melanomamodel, melanomamodel.val_loader, epoch)            #TODO Would this be better extracted outside of the train method?
            val_loss_bal, val_metrics_bal = validate(melanomamodel, melanomamodel.val_loader_balanced, epoch)

            if melanomamodel.scheduler is not None:
                if isinstance(melanomamodel.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                  melanomamodel.scheduler.step(val_loss)
                else:
                    melanomamodel.scheduler.step()

            print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
            print(f"    [Natural] Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")
            print(f"    [Balanced] Val Loss: {val_loss_bal:.4f}, Metrics: {val_metrics_bal}")

            wandb_val_log(avg_loss, val_loss, val_loss_bal, val_metrics, val_metrics_bal)

            savedmodel = save_checkpoint(melanomamodel.opt, melanomamodel.best_metrics, melanomamodel.model, epoch + 1, val_metrics)
            if savedmodel is not None:
                testmodels = [savedmodel]

    return testmodels


def train_batch(melanomamodel, images, labels, epoch):
    loss_function = melanomamodel.opt['model'].get('loss_function', 'bce').lower()
    contrastive_epochs = melanomamodel.opt['training'].get('contrastive_epochs', 5)
    use_contrastive = (loss_function == 'contrastive') and (epoch < contrastive_epochs)

    melanomamodel.optimizer.zero_grad()

    if not use_contrastive:
        images = extract_contrastive_pair(images)

    if use_contrastive:
        loss = train_contrastive_batch(melanomamodel, images, labels)

    else:

        images = images.to(melanomamodel.device)
        labels = labels.to(melanomamodel.device)

        # Important: Force normal model output (classifier head)
        preds = melanomamodel.model(images)

        loss_function = melanomamodel.opt['model'].get('loss_function', 'bce').lower()

        loss = melanomamodel.criterion(preds, labels.float())


    loss.backward()
    melanomamodel.optimizer.step()

    return loss



def validate(m, val_loader, epoch=1):
    device = m.device
    m.model.eval()
    total_loss = 0.0

    # Optional: early exit if contrastive validation applies
    avg_loss, metrics, early_exit = maybe_run_contrastive_validation(m, val_loader, epoch)
    if early_exit:
        return avg_loss, metrics

    bce_crit = nn.BCEWithLogitsLoss()
    all_outputs, all_labels = [], []

    with torch.no_grad():
        loop = tqdm(val_loader, desc="[Val]")
        for imgs, lbls in loop:
            if isinstance(imgs, (list, tuple)):
                imgs = imgs[0]
            imgs, lbls = imgs.to(device), lbls.to(device)

            preds = m.model(imgs)
            probs = torch.sigmoid(preds)
            loss = bce_crit(preds, lbls.float())
            total_loss += loss.item()

            all_outputs.append(probs.cpu())
            all_labels.append(lbls.cpu())

    avg_loss = total_loss / len(val_loader)
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    metrics = evaluate_metrics(m.opt, all_outputs, all_labels, epoch + 1)
    log_results(m.opt, metrics)

    return avg_loss, metrics



def validate_loss(melanomamodel, total_loss, val_loader, description="[Val]"):
    device = melanomamodel.device
    melanomamodel.model.eval()

    all_outputs = []
    all_labels = []

    loop = tqdm(val_loader, desc=description)
    with torch.no_grad():
        for images, labels in loop:
            labels = labels.to(device)

            if isinstance(images, (list, tuple)):
                images1, images2 = images
                images1, images2 = images1.to(device), images2.to(device)
                outputs, labels = evaluate_svm_outputs(melanomamodel, images1, images2, labels)
            else:
                images = images.to(device)
                outputs = melanomamodel.model(images)

            loss = melanomamodel.criterion(outputs, labels.float())
            total_loss += loss.item()

            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    return all_labels, all_outputs, total_loss


def test(opt, melanoma_model_list, val_loader, tag="natural"):
    if melanoma_model_list is None or len(melanoma_model_list) == 0:
        print("Test: No models to test. Exiting...")
        return

    predictonly = melanoma_model_list[0].predictmode
    print(f"Test: Generate predictions only = {predictonly}")

    output_list = []
    all_labels = None  # Capture once for all models

    for melanoma_test in melanoma_model_list:
        print(f"Test: Model {melanoma_test.model_path}")
        melanoma_test.model = melanoma_test.model.to(melanoma_test.device)
        melanoma_test.model.eval()

        total_loss = 0.0
        with torch.no_grad():
            labels, outputs, total_loss = validate_loss(melanoma_test, total_loss, val_loader, description='[Test]')

        # Capture labels once (they are same across all models)
        if all_labels is None:
            all_labels = labels

        if outputs.dim() > 1 and outputs.shape[1] == 1:
            outputs = outputs.squeeze(1)
        if labels.dim() > 1 and labels.shape[1] == 1:
            labels = labels.squeeze(1)

        output_list.append(outputs)

    # Ensemble outputs
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
