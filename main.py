'''
main.py - Contains train, val, test loop logic and model instantiation code
train and test executed with a <CONFIG>.yml file sourced from ./options folder. E.g.
    python main.py -o my_config.yml

test only with a saved model (relative path) via -s flag e.g.
    python main.py -o my_config.yml -s logs/<my_trained_model>/checkpoints/<my_saved_model>.pth
'''


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

# Builds a object that holds all the model, optimiser, dataloaders etc for a Test run
class MelanomaTest:
    def __init__(self, opt, testmodel):
        # instantiate the test model parameters
        self.opt = opt
        print(opt)
        # Set the device to GPU if available
        self.device = cuda_available(self.opt)

        # Instantiate the dataloaders for test
        self.predictmode, self.val_loader = melanoma_test_dataloaders(opt)
        # Instantiate a hybrid model or standard model as appropriate.
        if opt['model']['hybrid'].get('enabled', False):
            # check if the hybrid model is enabled in the config
            self.model = test_hybrid_model(opt, testmodel).to(self.device)
            print("Using Hybrid Model")
        else:
            # if the hybrid model is not enabled, use the standard model                                                 
            self.model = test_melanoma_model(opt, testmodel).to(self.device)
        # instantiate the model with the saved model path 
        self.model_path = testmodel 
        # Set the loss function based on YML config. 
        self.criterion = melanoma_loss(opt).to(self.device)

        # Create test metrics object to record output.
        self.best_metrics = {metric: float('-inf') for metric in opt['testing']['model_save_metrics']}

        self.logwandb = wandb_login(opt)
        # If wandb is enabled, log the model and config
        print("Wandb logging: ", self.logwandb)

        log_model(self.opt, self.model)

# Builds a object that holds all the model, optimiser, dataloaders etc for a Train and validation run
class MelanomaTrainer:
    """
    Class to hold the model, optimizer, dataloaders, and other training-related components.
    
    """
    def __init__(self, opt):
        # Instantiate the model parameters
        self.opt = opt
        # print the config options
        print(opt)
        # Set the device to GPU if available
        self.device = cuda_available(self.opt)
        # Make an contrasive object from the contrastive_svm module
        self.cengine = ContrastiveSVM(opt, self.device)
        # Sets either the k-fold dataloaders or the standard run dataloaders (train/val)
        if opt['dataset'].get('use_groupkfold', False):
            # K-FOLD dataloaders
            self.fold_loaders = melanoma_train_dataloaders(opt)
            self.is_kfold = True
            self.train_loader, self.val_loader, self.val_loader_balanced = None, None, None

        else:
            # Standard dataloaders
            # If the dataset is a balanced dataset, use the balanced dataloader
            self.train_loader, self.val_loader, self.val_loader_balanced = melanoma_train_dataloaders(opt)
            self.is_kfold = False
        # Instantiate a hybrid model or standard model as appropriate.
        if opt['model']['hybrid'].get('enabled', False):
            # check if the hybrid model is enabled in the config
            print("Using Hybrid Model")
            self.model = train_hybrid_model(opt).to(self.device)
        else:
            # if the hybrid model is not enabled, use the standard model
            self.model = train_melanoma_model(opt).to(self.device)
        # Define the loss function based on YML config. Uses loss.py helper methods.
        loss_fn = opt['model'].get('loss_function', 'bce').lower()
        if loss_fn == 'contrastive':
            # 1) supervised‐contrastive (or hinge) during the pre‐training phase
            self.criterion = melanoma_loss(opt, self.train_loader).to(self.device)

            # 2) plain BCE for the fine‐tune / classifier head
            bce_opt = deepcopy(opt)
            # set the loss function to bce for the second loss
            bce_opt['model']['loss_function'] = 'bce'
            self.criterion_second = melanoma_loss(bce_opt, self.train_loader).to(self.device)
        else:
            # single‐loss-mode
            self.criterion = melanoma_loss(opt, self.train_loader).to(self.device)

        # Finalise optimiser, scheduler and metrics config for the model being built
        self.optimizer = self.get_optimizer()
        # Set the scheduler based on the config
        self.scheduler = self.get_scheduler()
        # Set the scaler for mixed precision training
        self.scaler = amp.GradScaler() if opt['training']['mixed_precision'] else None
        # reset the best metrics for the model
        self.best_metrics = self.reset_metrics()

        # if the config indicates to freeze the backbone, set the freeze_backbone_epochs and freeze_backbone
        self.freeze_backbone_epochs = opt['training'].get('freeze_backbone_epochs', 0)
        
        # if the config indicates to freeze the backbone, set the freeze_backbone
        self.backbone_frozen = opt['training'].get('freeze_backbone', True)

        # Define the feature loss function based on the config
        # If the loss function is contrastive or triplet and not combined, set the use_feature_loss to True
        use_feature_loss = (
        (self.opt['model'].get('loss_function','').lower() == 'contrastive')
        or
        (self.opt['model'].get('loss_function','').lower() == 'triplet'
        and not self.opt['model'].get('combined_loss', False))
        )
        # Contrastive learning freezing logic
        if self.backbone_frozen and not use_feature_loss:
            print(f"Freezing backbone for {self.freeze_backbone_epochs} epochs (classification-only).")
            self.freeze_backbone(True)
        elif self.backbone_frozen and use_feature_loss:
            # We _need_ gradients for triplet/contrastive!
            print(" Unfreezing backbone for feature-based loss training.")
            self.freeze_backbone(False)
        else:
            # never requested to freeze at all
            print("Backbone will not be frozen.")

        self.logwandb = wandb_login(opt)
        print("Wandb: ", self.logwandb)

        # Write model to the logs folder for posterity
        log_model(self.opt, self.model)

    def reset_metrics(self):
        return {metric: float('-inf') for metric in self.opt['testing']['model_save_metrics']}

    # Optimizer loading code based on config
    def get_optimizer(self):
        # set the optimizer based on the config
        if self.opt['training']['optimizer'] == 'adam':
            # Adam optimizer with weight decay
            return optim.Adam(self.model.parameters(), lr=self.opt['training']['learning_rate'])
        elif self.opt['training']['optimizer'] == 'sgd':
            # SGD optimizer with momentum
            return optim.SGD(self.model.parameters(), lr=self.opt['training']['learning_rate'], momentum=0.9)
        elif self.opt['training']['optimizer'] == 'adamw':
            # AdamW optimizer with weight decay
            return optim.AdamW(self.model.parameters(), lr=self.opt['training']['learning_rate'])
        elif self.opt['training']['optimizer'] == 'adagrad':
            # Adagrad optimizer with weight decay
            return optim.Adagrad(self.model.parameters(), lr=self.opt['training']['learning_rate'])
        elif self.opt['training']['optimizer'] == 'amsgrad':
            # Adam optimizer with AMSGrad
            return optim.Adam(self.model.parameters(), lr=self.opt['training']['learning_rate'], amsgrad=True)

    # Set scheduler from config
    def get_scheduler(self):
        # Set the scheduler based on the config
        if self.opt['training']['scheduler'] == 'cosine':
            # Cosine annealing scheduler
            # T_max is the number of epochs
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt['training']['epochs'])
        elif self.opt['training']['scheduler'] == 'step':
            # Step scheduler with step size and decay rate
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[self.opt['training']['step_size']],
                gamma=self.opt['training']['decay_rate']
            )
        elif self.opt['training']['scheduler'] == 'reduce_on_plateau':
            # Reduce learning rate on plateau
            # patience is the number of epochs with no improvement
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1, verbose=True)
        else:
            return None

    def freeze_backbone(self, freeze: bool = False):
        # 1) Freeze / unfreeze the backbone
        for p in self.model.backbone.parameters():
            p.requires_grad = not freeze
    
        # 2) Always keep the heads trainable
        for head in (self.model.classifier, self.model.svm_head, self.model.projector):
            if head is not None:
                for p in head.parameters():
                    p.requires_grad = True

        print(f"Backbone layers frozen?= {freeze}")

    #TODO Work out why this exists when most is called on class instatiation? Only used in KFOLD code?
    def setup_training(self, train_loader):
        """Common initializer for criterion, optimizer, scheduler, scaler."""
        # instantiate the lost function based on the config
        lf = self.opt['model'].get('loss_function', 'bce').lower()
    
        if lf == 'contrastive':
            # 1) supervised‐contrastive (or hinge) during the pre‐training phase
            self.criterion = melanoma_loss(self.opt, train_loader).to(self.device)
    
            # BCE loss for the fine-tune/classifier head
            bce_opt = deepcopy(self.opt)
            bce_opt['model']['loss_function'] = 'bce'
            self.criterion_second = melanoma_loss(bce_opt, train_loader).to(self.device)
        else:

            # single‐loss-mode
            self.criterion = melanoma_loss(self.opt, train_loader).to(self.device)

        # Set the loss function based on the config
        self.optimizer = self.get_optimizer()
        # Set the scheduler based on the config
        self.scheduler = self.get_scheduler()
        # Set the scaler for mixed precision training
        self.scaler = amp.GradScaler() if self.opt['training']['mixed_precision'] else None
        # Set the scaler for mixed precision training
        self.best_metrics = {metric: float('-inf') for metric in self.opt['testing']['model_save_metrics']}
    
        print(f"Training setup complete with loss function: {lf}")

def train(melanomamodel):
    """
    Main training loop. Handles both k-fold and standard training.
    Most of the modules are called from the MelanomaTrainer class.

    """
    print("Starting Training")
    wandb_watch(melanomamodel.model, melanomamodel.criterion, log_freq=10)
    testmodels = []

    # K-FOLD loop
    if melanomamodel.is_kfold:
        # Loop through each fold in the k-fold dataloaders
        for fold_data in melanomamodel.fold_loaders:
            # Unpack the fold data
            fold_idx = fold_data['fold']
            # Set the dataloaders for the current fold
            print(f"\n[INFO] Starting Fold {fold_idx}")

            # Uses a model PER fold rather than a global model. Instantiated here.
            melanomamodel.model = train_melanoma_model(melanomamodel.opt).to(melanomamodel.device)
            # reset the metrics for the model
            melanomamodel.best_metrics = melanomamodel.reset_metrics()
            # Set the train dataloaders for the current fold
            train_loader = fold_data['train_loader']
            # Set the avalidation dataloaders for the current fold
            val_loader = fold_data['val_loader']
            # Set the balanced validation dataloaders for the current fold
            val_loader_balanced = fold_data['val_loader_balanced']

            # reinitialise the model and optimisers for the current fold
            melanomamodel.setup_training(train_loader)

            wandb_watch(melanomamodel.model, melanomamodel.criterion, log_freq=10)
            modeltokeep = None

            # Loop trainng for each epoch for the current fold
            for epoch in range(melanomamodel.opt['training']['epochs']):
                # set the model to training mode
                melanomamodel.model.train()
                total_loss = 0.0
                # Batching
                # Loop through the training dataloader for the current fold
                loop = tqdm(train_loader, desc=f"[Fold {fold_idx}] Epoch {epoch+1}/{melanomamodel.opt['training']['epochs']}")
                # Loop through the training dataloader for the current fold
                for images, labels in loop:
                    # calculate the loss for the current batch by calling the train_batch function
                    loss = train_batch(melanomamodel, images, labels, epoch)
                    total_loss += loss.item()
                    loop.set_postfix(loss=loss.item())
                # check if the model is in contrastive mode and call the contrastive engine
                melanomamodel.cengine.on_contrastive_phase_end(melanomamodel, epoch)

                # Validatation loop for the fold epoch
                avg_loss = total_loss / len(train_loader)
                val_loss, val_metrics = validate(melanomamodel, val_loader, epoch, tag='natural')
                # Calculates validation metrics for both natural and balanced test sets.
                val_loss_bal, val_metrics_bal = validate(melanomamodel, val_loader_balanced, epoch, tag='balanced')

                # Update the optimiser if a scheduler is configured.
                if melanomamodel.scheduler:
                    if isinstance(melanomamodel.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        melanomamodel.scheduler.step(val_loss)
                    else:
                        # Step the scheduler
                        melanomamodel.scheduler.step()

                print(f"[Fold {fold_idx}] Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
                print(f"    [Natural] Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")
                print(f"    [Balanced] Val Loss: {val_loss_bal:.4f}, Balanced Metrics: {val_metrics_bal}")

                wandb_val_log(avg_loss, val_loss, val_loss_bal, val_metrics, val_metrics_bal)

                # strip contrastive components if needed (unchanged)
                
                # Save our checkpoint model. Note checkpoint saving is based on metric and save approach (e.g. 'best auc')
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

            # Keep the file name of the model to be used in the test loop (if the current epoch is better)
            if modeltokeep:
                testmodels.append(modeltokeep)

    # STANDARD TRAINING LOOP (not KFOLD)
    else:
        # if the model is not in kfold mode, set the dataloaders to the standard dataloaders
        for epoch in range(melanomamodel.opt['training']['epochs']):
            # Set the model to training mode
            melanomamodel.model.train()
            total_loss = 0

            # Batching
            # Loop through the training dataloader
            loop = tqdm(melanomamodel.train_loader, desc=f"Epoch {epoch+1}/{melanomamodel.opt['training']['epochs']}")
            # Loop through the training dataloader
            for images, labels in loop:
                # calculate the loss for the current batch by calling the train_batch function
                loss = train_batch(melanomamodel, images, labels, epoch)    
                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            # check if the model is in contrastive mode and call the contrastive engine
            melanomamodel.cengine.on_contrastive_phase_end(melanomamodel, epoch)    

            # calculate the average loss for the current epoch
            avg_loss = total_loss / len(melanomamodel.train_loader)
            # Calculates validation metrics for both natural and balanced test sets.
            val_loss, val_metrics = validate(melanomamodel, melanomamodel.val_loader, epoch, tag='natural')
            val_loss_bal, val_metrics_bal = validate(melanomamodel, melanomamodel.val_loader_balanced, epoch, tag='balanced')

            if melanomamodel.scheduler:
                # Step the scheduler
                # if the scheduler is ReduceLROnPlateau, step the scheduler with the validation loss
                if isinstance(melanomamodel.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    melanomamodel.scheduler.step(val_loss)
                else:
                    # if the scheduler is not ReduceLROnPlateau, step the scheduler
                    melanomamodel.scheduler.step()

            # Print the training and validation loss for the current epoch
            print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}")
            # Print the validation loss and metrics for the Natural test set
            print(f"    [Natural] Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")
            # Print the validation loss and metrics for the balanced test set
            print(f"    [Balanced] Val Loss: {val_loss_bal:.4f}, Metrics: {val_metrics_bal}")

            wandb_val_log(avg_loss, val_loss, val_loss_bal, val_metrics, val_metrics_bal)

           
            # Save our checkpoint model. Note checkpoint saving is based on the best AUC and save approach.
            # Save the model if the current epoch is better than the previous best epoch
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

# Resuable code block for training a batch.
def train_batch(melanomamodel, images, labels, epoch):
    """
    Train a single batch of data.

    """
    # Read the loss function from the config
    lf = melanomamodel.opt['model'].get('loss_function','bce').lower()
    # If the combined loss is enabled, identify it.
    combined = melanomamodel.opt['model'].get('combined_loss', False)
    # Read the second loss function from the config
    second   = melanomamodel.opt['model'].get('second_loss', '').lower()
    # If contrastive is enabled, and the epoch is less than the contrastive epochs, call the contrastive engine
    if lf == 'contrastive' and epoch < melanomamodel.opt['training']['contrastive_epochs']:
        return melanomamodel.cengine.train_batch(melanomamodel, images, labels, epoch)

    # Zero our gradients
    melanomamodel.optimizer.zero_grad()
    # If the use metadata flag is set in the config, use the metadata
    if bool(melanomamodel.opt['model'].get('use_metadata', False)):
        # Unpack the images and metadata
        img, meta = images
        # Move the images and metadata to the device
        img  = img.to(melanomamodel.device)
        meta = meta.to(melanomamodel.device)
        images = (img, meta) 
    else:
        # If images is a tuple/list, drop the second element
        img = images if not isinstance(images, (list, tuple)) else images[0]
        images = img.to(melanomamodel.device)

    # Move the labels to the device
    labels = labels.to(melanomamodel.device)


    if lf == 'contrastive':
        # 1) supervised‐contrastive (or hinge) during the pre‐training phase
        preds = melanomamodel.model(images, return_projection=True)
    elif lf == 'triplet':
        # 2) triplet loss during the pre‐training phase
        preds = melanomamodel.model(images, return_features=True)
    elif combined and second in ['triplet', 'contrastive']:
        # 3) if the combined loss is enabled and the second loss is triplet or contrastive, return the logits and features
        preds = melanomamodel.model(images,
                                    return_features=True,
                                    return_logits=True)
    # 4) All other cases (e.g. BCE or BCE+Dice): just logits
    else:
        # 5) standard forward pass
        preds = melanomamodel.model(images)
    
    # pass the pres and lables to the loss function
    loss = melanomamodel.criterion(preds, labels.float())
    # do backpropagation
    loss.backward()
    melanomamodel.optimizer.step()

    return loss

# Validation loop code (usually called at the end of a training epoch)
def validate(m, val_loader, epoch=1, tag='notag'):
    """
    Wrapper that routes to contrastive or standard validation.
    """
    # Read the loss function from the config
    lf = m.opt['model']['loss_function']
    # Check loss function is contrastive and the epoch is less than the contrastive epochs
    if lf == 'contrastive' and epoch < m.opt['training']['contrastive_epochs']:
        # Call the contrastive engine for validation
        return m.cengine.validate(m, val_loader, epoch)

    device = m.device
    # Set the model to evaluation mode
    m.model.eval()
    total_loss = 0.0
    # Set the loss function to BCE with logits
    bce_crit = nn.BCEWithLogitsLoss()
    all_outputs, all_labels = [], []

    # Start the validation loop by disabling gradients
    with torch.no_grad():
        # Loop through the validation dataloader
        loop = tqdm(val_loader, desc="[Val]")
        for images, labels in loop:
            # If the use metadata flag is set in the config, use the metadata
            if bool(m.opt['model'].get('use_metadata', False)):
                img, meta = images
                img  = img.to(device)
                meta = meta.to(device)
                images = (img, meta) 
            else:
                # If images is a tuple/list, drop the second element
                img = images if not isinstance(images, (list, tuple)) else images[0]
                # Move the images to the device
                images = img.to(device)
            # Move the labels to the device
            labels = labels.to(device)
            # Pass the images through the model
            # Get the logits from the model
            preds = m.model(images)
            # Transform the logits to probabilities
            probs = torch.sigmoid(preds)
            # calculate the loss
            loss = bce_crit(preds, labels.float())
            total_loss += loss.item()
            # Pass the logits and labels to the all_outputs and all_labels lists
            all_outputs.append(probs.cpu())
            all_labels.append(labels.cpu())
            
    # calculate the average loss for the current epoch
    avg_loss = total_loss / len(val_loader)
    # Convert the all_outputs and all_labels lists to tensors
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    # Get the results from the metrics function
    metrics = evaluate_metrics(m.opt, all_outputs, all_labels, epoch+1, tag=tag)
    log_results(m.opt, metrics, phase='val', tag=tag)

    return avg_loss, metrics

# Generates predictions for test loop. Called from test()
def test_outputs(melanomamodel, total_loss, val_loader, description="[Val]"):
    """
    Wrapper for contrastive or standard test.
    """
    # Read the loss function from the config and check if it is contrastive
    if melanomamodel.opt['model']['loss_function'] == 'contrastive':
        # Call the contrastive engine for validation
        return melanomamodel.cengine.validate_loss(melanomamodel, total_loss, val_loader, description)

    # Pass the model to the device
    device = melanomamodel.device
    # Set the model to evaluation mode
    melanomamodel.model.eval()
    all_outputs, all_labels = [], []
    loop = tqdm(val_loader, desc=description)

    # Start the validation loop by disabling gradients
    with torch.no_grad():
        # Loop through the validation dataloader
        for images, labels in loop:
            if bool(melanomamodel.opt['model'].get('use_metadata', False)):
                # If the use metadata flag is set in the config, use the metadata
                # Unpack the images and metadata
                img, meta = images           
                img  = img.to(device)
                meta = meta.to(device)
                images = (img, meta)
            else:
                # images may be a tuple (img,meta) or a single Tensor
                img = images[0] if isinstance(images, (list,tuple)) else images
                # Move the images to the device
                images = img.to(device)
            # Move the labels to the device
            labels = labels.to(device)
            outputs = melanomamodel.model(images)
            # Calculate the loss
            loss = melanomamodel.criterion(outputs, labels.float())
            total_loss += loss.item()
            # Append the outputs and labels to the all_outputs and all_labels lists
            all_outputs.append(outputs.cpu())
            all_labels.append(labels.cpu())

    # Concatenate the all_outputs and all_labels lists to tensors
    all_outputs = torch.cat(all_outputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_labels, all_outputs, total_loss

# Executes the test loop.
# Note that this supports both single and ensemble model testing (k-fold)
def test(opt, melanoma_model_list, val_loader, tag="notag"):
    """
    Wrapper to route to contrastive or standard test.
    """
    # Read the loss function from the config and check if it is contrastive
    if opt['model']['loss_function'] == 'contrastive':
        # Call the contrastive engine for validation
        return melanoma_model_list[0].trainer.cengine.test(
           opt, melanoma_model_list, val_loader, tag)

    if melanoma_model_list is None or len(melanoma_model_list) == 0:
        # If there are no models to test, exit
        print("Test: No models to test. Exiting...")
        return

    # If we want to predict only, set the predictonly flag to True
    predictonly = melanoma_model_list[0].predictmode
    print(f"Test: Generate predictions only = {predictonly}")

    output_list = []
    all_labels = None

    # Loop for each model to test (supports both standard and k-fold)
    for melanoma_test in melanoma_model_list:
        print(f"Test: Model {melanoma_test.model_path}")
        # pass the model to the device
        melanoma_test.model = melanoma_test.model.to(melanoma_test.device)
        # Set the model to evaluation mode
        melanoma_test.model.eval()

        total_loss = 0.0
        # Define the evalution phase by disabling gradients
        with torch.no_grad():
            # Run the test predictions for the current model
            labels, outputs, total_loss = test_outputs(melanoma_test, total_loss, val_loader, description='[Test]')
        # If the list of labels is empty, set the all_labels to the current labels
        if all_labels is None:
            all_labels = labels

        if outputs.dim() > 1 and outputs.shape[1] == 1:
            # If the outputs are 2D and the second dimension is 1, squeeze the outputs
            outputs = outputs.squeeze(1)
        if all_labels.dim() > 1 and all_labels.shape[1] == 1:
            all_labels = all_labels.squeeze(1)

        output_list.append(outputs)

    # This call to soft-voting is agnostic of k-fold or single models.
    # In the case of single models it is a single dimension input and returns the probs without adjustment (x / 1 = x).
    # In the case of ensemble the stack is multi-dimensional and soft voting returns the mean probability for each test position across the models
    ensemble_logits = torch.stack(output_list, dim=0)
    probabilities = soft_voting_probs_from_logits(ensemble_logits)

    # Predict only supports predictions without GT labels (kaggle comp)
    if predictonly:
        print("Saving predictions only (no ground truth labels available).")
        write_kaggle_csv(opt, val_loader.dataset.files, probabilities, tag=tag)
    else:
        print("Evaluating test metrics...")
        metrics = evaluate_metrics(opt, probabilities, all_labels, epoch="Test", tag=tag)
        log_test(opt, metrics, tag=tag)
        wandb_test_log(metrics, tag=tag)
        print(f"Test Metrics ({tag}): {metrics}")

# Parses the YML file provided at command line
def argument_parser():
    """
    Parses the command line arguments and loads the YML config file.
    
    """
    # Set the default config file to default.yml
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
    # set the seed for random number generation
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Tradeoff: slows down but ensures deterministic behavior

# Main method.
# Calls training or test loops as appropriate.
def main():
    set_seed(42)
    # Parse the command line arguments and load the YML config file
    opt =  argument_parser()
    # Test models is a list of saved model paths.
    testmodels = opt['dataset']['savedmodel']   

    # Check to see if we should train first
    if testmodels is None or len(testmodels) == 0:         
        print("TRAIN MODEL MODE")
        # If no saved model is provided, train the model
        melanomamodel = MelanomaTrainer(opt)
        testmodels = train(melanomamodel)
        # If the testmodels is empty, exit
        if testmodels is None or len(testmodels) == 0:
            print("Training complete. No Saved Model. Exiting.")
            return          # Nothing to test

    # Ensure feature-based modes are turned off during testing
    if (
        opt['model']['loss_function'] in ('contrastive', 'triplet')
        or opt['model'].get('combined_loss', False)
    ):
        print(
            " Disabling feature-based loss for test phase; "
            f"running pure 'bce' instead of "
            f"'{opt['model'].get('loss_function', '')}'"
            + (", combined" if opt['model'].get('combined_loss') else "")
        )
        opt['model']['loss_function'] = 'bce'
        # drop any secondary triplet/focal/etc.
        opt['model']['combined_loss'] = False
        opt['model']['second_loss'] = None
        opt['model']['mode'] = 'regular'

    # Test Loop begins
    melanomatests = []
    for model in testmodels:
        melanomatests.append(MelanomaTest(opt, model))

    print("=== Natural test ===")
    test(opt, melanomatests, melanomatests[0].val_loader, tag="natural")

    if opt['dataset'].get('dataset_balanced_test_csv'):
        print("\n=== Balanced test ===")
        opt['dataset']['dataset_test_csv'] = opt['dataset']['dataset_balanced_test_csv']
        balanced_tests = [MelanomaTest(opt, mt.model_path) for mt in melanomatests]
        test(opt, balanced_tests, balanced_tests[0].val_loader, tag="balanced")


if __name__ == "__main__":
    main()
