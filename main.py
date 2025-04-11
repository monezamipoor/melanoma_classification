import argparse
import yaml
import os
import torch
import torch.cuda.amp as amp
from torch import optim
from torchinfo import summary
from tqdm import tqdm

import utils
from data import melanoma_dataloaders
from model import melanoma_model
from loss import melanoma_loss
from utils import log_results, cuda_available, log_model, save_checkpoint
from metrics import evaluate_metrics
from datetime import datetime
from wandb_helper import wandb_login, wandb_watch, wandb_train_log, wandb_val_log
import wandb
import numpy as np
import matplotlib.pyplot as plt

wandb.init(mode="disabled")


# TODO comments needed
def denormalize_image(tensor, mean, std):

    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

# TODO save_dir needs to be parameterised (Ashkan). Refactor to separate class
def save_augmented_samples(loader, num_samples=10, save_dir="/content/drive/MyDrive/melanoma_classification/logs/Sample"):
      
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Get one batch from the DataLoader. Assuming batch[0] contains the images.
    batch = next(iter(loader))
    images = batch[0]  # (B, C, H, W)

    # Define the normalization parameters used in your transforms:
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    imgs_denorm = []
    for i in range(num_samples):
        img = images[i].clone().cpu()
        img = denormalize_image(img, imagenet_mean, imagenet_std)
        # Convert from (C, H, W) to (H, W, C)
        img_np = img.permute(1, 2, 0).numpy()
        # Clip values to [0, 1] for display purposes
        img_np = np.clip(img_np, 0, 1)
        imgs_denorm.append(img_np)

    # Create a grid plot for the samples
    fig, axes = plt.subplots(1, num_samples, figsize=(20, 5))
    for idx, ax in enumerate(axes):
        ax.imshow(imgs_denorm[idx])
        ax.set_title(f"Sample {idx+1}")
        ax.axis("off")
    plt.suptitle("Augmented Samples")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_path = os.path.join(save_dir, f"augmented_samples_{timestamp}.png")

    # Save the figure to the unique file path
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved augmented samples to {save_path}")


class MelanomaTrainer:
    def __init__(self, opt):
        self.opt = opt
        print(opt)
        self.device = cuda_available(self.opt)
        # K-Fold 
        if opt['dataset'].get('use_groupkfold', False):
            # Expecting that melanoma_dataloaders() returns a list of dicts for each fold
            self.fold_loaders = melanoma_dataloaders(opt)  # e.g. [{'fold': 0, 'train_loader': ..., 'val_loader': ...}, ...]
            self.is_kfold = True
        else:
            self.train_loader, self.val_loader = melanoma_dataloaders(opt)
            self.is_kfold = False

        self.model = melanoma_model(opt).to(self.device)
        self.criterion = melanoma_loss(opt).to(self.device)
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.scaler = amp.GradScaler() if opt['training']['mixed_precision'] else None
        self.best_metrics = {metric: float('-inf') for metric in opt['testing']['model_save_metrics']}

        '''if opt['training']['freeze_pretrained']:
            self.freeze_backbone(bool(opt['training']['freeze_pretrained']))
        else:
            self.freeze_backbone(False)'''

        self.logwandb = wandb_login(opt)  # Track if we have an active wandb login
        print("Wandb: ", self.logwandb)

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
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt['training']['step_size'],
                                                   gamma=self.opt['training']['decay_rate'])
        elif self.opt['training']['scheduler'] == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1, verbose=True)
        else:
            return None

    def freeze_backbone(self, freeze=False):
        for param in list(self.model.parameters())[:-1]:
            param.requires_grad = not freeze
        print("Backbone layers frozen?= " + str(freeze))

    def train(self):
        #summary(self.model, input_size=(1, 3, 224, 224))       # Quick print of model arch if needed
        print("Starting Training")
        wandb_watch(self.model, self.criterion, log_freq=10)

        if self.is_kfold:
            for fold_data in self.fold_loaders:
                fold_idx = fold_data['fold']
                print(f"\n[INFO] Starting Fold {fold_idx}")

                # Re-initialize the model for each fold (bassically a fresh start):
                self.model = melanoma_model(self.opt).to(self.device)
                self.optimizer = self.get_optimizer()
                self.scheduler = self.get_scheduler()

                train_loader = fold_data['train_loader']
                val_loader   = fold_data['val_loader']

                wandb_watch(self.model, self.criterion, log_freq=10)

                for epoch in range(self.opt['training']['epochs']):
                    self.model.train()
                    total_loss = 0

                    loop = tqdm(train_loader, desc=f"[Fold {fold_idx}] Epoch {epoch+1}/{self.opt['training']['epochs']}")

                    for images, labels in loop:
                        loss = self.train_batch(images, labels)

                        if self.opt['training']['gradient_clipping']:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['training']['gradient_clipping'])

                        total_loss += loss.item()
                        loop.set_postfix(loss=loss.item())

                    # Log final batch loss for the epoch
                    wandb_train_log(epoch+1, float(loss))

                    avg_loss = total_loss / len(train_loader)

                    # Validate on this fold's val loader
                    val_loss, val_metrics = self.validate(val_loader, epoch)

                    # Step the scheduler if applicable
                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(val_loss)
                        else:
                            self.scheduler.step()

                    print(f"[Fold {fold_idx}] Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

                    # Log validation results to wandb
                    wandb_val_log(avg_loss, val_loss, **val_metrics,)

                    # Save checkpoint for best model or last, etc.
                    save_checkpoint(self.opt, self.best_metrics, self.model, epoch+1, val_metrics)

        else:
            # Single train/val scenario
            print("Starting Training")
            wandb_watch(self.model, self.criterion, log_freq=10)

            for epoch in range(self.opt['training']['epochs']):
              self.model = self.model.to(self.device)
              self.model.train()
              total_loss = 0

              loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.opt['training']['epochs']}")

              #If you want to see the images after Aug
              # save_augmented_samples(train_loader_p, num_samples=10, save_dir="/content/drive/MyDrive/melanoma_classification/logs/Sample")

              for images, labels in loop:
                  loss = self.train_batch(images, labels)

                  if self.opt['training']['gradient_clipping']:
                      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['training']['gradient_clipping'])

                  total_loss += loss.item()
                  loop.set_postfix(loss=loss.item())


              wandb_train_log(epoch+1, float(loss))

              avg_loss = total_loss / len(self.train_loader)
              val_loss, val_metrics = self.validate(self.val_loader, epoch)            #TODO Would this be better extracted outside of the train method?

              if self.scheduler is not None:
                  self.scheduler.step(val_loss if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)

              print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

              wandb_val_log(avg_loss, val_loss, **val_metrics)

              save_checkpoint(self.opt, self.best_metrics, self.model, epoch + 1, val_metrics)


    def train_batch(self, images, labels):
        images, labels = images.to(self.device), labels.to(self.device)
        self.optimizer.zero_grad()
        # TODO mixed precision is not tested
        if self.opt['training']['mixed_precision']:
            with amp.autocast():
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            outputs = self.model(images)
            loss = self.criterion(outputs.squeeze(1),
                                  labels.float())  # Need to squeeze [BS, 1] to [BS] and BCE uses float
            loss.backward()
            self.optimizer.step()
        return loss

    def validate(self, val_loader, epoch):
        self.model = self.model.to(self.device)
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            loop = tqdm(val_loader, desc="[Val]")

            firstitr = True
            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs.view(-1), labels.view(-1).float())

                total_loss += loss.item()

                if firstitr:
                    all_outputs = outputs.cpu()
                    all_labels  = labels.cpu()
                    firstitr = False
                else:
                    all_outputs = torch.cat((all_outputs, outputs.cpu()), dim=0)
                    all_labels  = torch.cat((all_labels, labels.cpu()), dim=0)

        avg_loss = total_loss / len(self.val_loader)
        metrics = evaluate_metrics(self.opt, all_outputs.squeeze(1), all_labels, epoch+1)
        log_results(self.opt, metrics)
        return avg_loss, metrics


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt", type=str, default="default.yml", help="the option file")
    args = parser.parse_args()
    if not os.path.isabs(args.opt) and not args.opt.startswith('./'):
        args.opt = os.path.join("./options", args.opt)
    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    opt['opt'] = args.opt
    return opt

def main():
    opt = argument_parser()
    trainer = MelanomaTrainer(opt)
    trainer.train()

if __name__ == "__main__":
    main()
