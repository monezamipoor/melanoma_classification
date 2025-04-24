import argparse
import yaml
import os
import torch
import torch.cuda.amp as amp
from torch import optim
from torchinfo import summary
from tqdm import tqdm

import utils

from data import melanoma_train_dataloaders, melanoma_test_dataloaders
from model import melanoma_model
from loss import melanoma_loss
from utils import log_results, cuda_available, log_model, save_checkpoint, write_kaggle_csv
from metrics import evaluate_metrics
from datetime import datetime
from wandb_helper import wandb_login, wandb_watch, wandb_train_log, wandb_val_log
import wandb
import numpy as np
import matplotlib.pyplot as plt

#wandb.init(mode="disabled")



import math
import numpy as np




def denormalize_image(tensor, mean, std):
    """Undo ImageNet normalization on a single C×H×W tensor."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def save_augmented_samples(loader, num_samples=10, save_dir=None, ncols=10):
    # 1) collect exactly num_samples images
    imgs = []
    for batch in loader:
        batch_imgs = batch[0]    # assumes loader yields (images, labels)
        for img in batch_imgs:
            imgs.append(img.clone().cpu())
            if len(imgs) >= num_samples:
                break
        if len(imgs) >= num_samples:
            break

    if not imgs:
        print("  No images found in loader!")
        return
    imgs = imgs[:num_samples]

    # 2) denormalize each
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]
    imgs_denorm = []
    for img in imgs:
        img = denormalize_image(img, imagenet_mean, imagenet_std)
        img_np = img.permute(1, 2, 0).numpy()
        imgs_denorm.append(np.clip(img_np, 0, 1))

    # 3) set up grid
    nrows = math.ceil(num_samples / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx < len(imgs_denorm):
            ax.imshow(imgs_denorm[idx])
        ax.axis("off")
    plt.suptitle("Augmented Samples", y=1.02)

    # 4) save or show
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(save_dir, f"augmented_samples_{timestamp}.png")
        plt.savefig(save_path, bbox_inches='tight')
        plt.close(fig)
        print(f"✅ Saved augmented samples to {save_path}")
    else:
        plt.show()
        plt.close(fig)




class MelanomaTest:
    def __init__(self, opt):
        self.opt = opt
        print(opt)
        self.device = cuda_available(self.opt)

        self.predictmode, self.val_loader = melanoma_test_dataloaders(opt)
        self.is_kfold = False

        self.model = melanoma_model(opt).to(self.device)
        self.criterion = melanoma_loss(opt).to(self.device)
        self.best_metrics = {metric: float('-inf') for metric in opt['testing']['model_save_metrics']}

        self.logwandb = wandb_login(opt)  # Track if we have an active wandb login
        print("Wandb: ", self.logwandb)

        log_model(self.opt, self.model)

class MelanomaTrainer:
    def __init__(self, opt):
        self.opt = opt
        print(opt)
        self.device = cuda_available(self.opt)
        # K-Fold 
        if opt['dataset'].get('use_groupkfold', False):
            # Expecting that melanoma_dataloaders() returns a list of dicts for each fold

            # print(f"\n Train Loader Size: {len(self.train_loader.dataset)} samples")
            # print(f" Val Loader Size: {len(self.val_loader.dataset)} samples")

            # # Optional: Print label distribution in train set
            # targets = [label for _, label in self.train_loader.dataset]
            # if torch.is_tensor(targets[0]):
            #     targets = [t.item() for t in targets]
            # print(f"🔍 Train Labels Distribution: {np.bincount(np.array(targets).astype(int))}")


            self.fold_loaders = melanoma_train_dataloaders(opt)  # e.g. [{'fold': 0, 'train_loader': ..., 'val_loader': ...}, ...]
            self.is_kfold = True
        else:
            self.train_loader, self.val_loader = melanoma_train_dataloaders(opt)
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

        log_model(self.opt, self.model)


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

    if melanomamodel.is_kfold:
        for fold_data in melanomamodel.fold_loaders:
            fold_idx = fold_data['fold']
            print(f"\n[INFO] Starting Fold {fold_idx}")

            # Re-initialize the model for each fold (bassically a fresh start):
            melanomamodel.model = melanoma_model(melanomamodel.opt).to(melanomamodel.device)
            melanomamodel.optimizer = melanomamodel.get_optimizer()
            melanomamodel.scheduler = melanomamodel.get_scheduler()

            train_loader = fold_data['train_loader']
            val_loader   = fold_data['val_loader']

            wandb_watch(melanomamodel.model, melanomamodel.criterion, log_freq=10)

            for epoch in range(melanomamodel.opt['training']['epochs']):
                melanomamodel.model.train()
                total_loss = 0

                loop = tqdm(train_loader, desc=f"[Fold {fold_idx}] Epoch {epoch+1}/{melanomamodel.opt['training']['epochs']}")

                for images, labels in loop:
                    loss = train_batch(melanomamodel, images, labels)

                    if melanomamodel.opt['training']['gradient_clipping']:
                        torch.nn.utils.clip_grad_norm_(melanomamodel.model.parameters(), melanomamodel.opt['training']['gradient_clipping'])

                    total_loss += loss.item()
                    loop.set_postfix(loss=loss.item())

                # Log final batch loss for the epoch
                wandb_train_log(epoch+1, float(loss))

                avg_loss = total_loss / len(train_loader)

                # Validate on this fold's val loader
                val_loss, val_metrics = validate(melanomamodel, val_loader, epoch)

                # Step the scheduler if applicable
                if melanomamodel.scheduler is not None:
                    if isinstance(melanomamodel.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        melanomamodel.scheduler.step(val_loss)
                    else:
                        melanomamodel.scheduler.step()

                print(f"[Fold {fold_idx}] Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

                # Log validation results to wandb
                wandb_val_log(avg_loss, val_loss, **val_metrics,)

                # Save checkpoint for best model or last, etc.
                save_checkpoint(melanomamodel.opt, melanomamodel.best_metrics, melanomamodel.model, epoch + 1, val_metrics)

    else:
        # Single train/val scenario
        print("Starting Training")
        wandb_watch(melanomamodel.model, melanomamodel.criterion, log_freq=10)

        for epoch in range(melanomamodel.opt['training']['epochs']):
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
              loss = train_batch(melanomamodel, images, labels)

              if melanomamodel.opt['training']['gradient_clipping']:
                  torch.nn.utils.clip_grad_norm_(melanomamodel.model.parameters(), melanomamodel.opt['training']['gradient_clipping'])

              total_loss += loss.item()
              loop.set_postfix(loss=loss.item())


          wandb_train_log(epoch+1, float(loss))

          avg_loss = total_loss / len(melanomamodel.train_loader)
          val_loss, val_metrics = validate(melanomamodel, melanomamodel.val_loader, epoch)            #TODO Would this be better extracted outside of the train method?

          if melanomamodel.scheduler is not None:
              melanomamodel.scheduler.step(val_loss if isinstance(melanomamodel.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)

          print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

          wandb_val_log(avg_loss, val_loss, **val_metrics)

          save_checkpoint(melanomamodel.opt, melanomamodel.best_metrics, melanomamodel.model, epoch + 1, val_metrics)


def train_batch(melanomamodel, images, labels):
    images, labels = images.to(melanomamodel.device), labels.to(melanomamodel.device)
    melanomamodel.optimizer.zero_grad()
    # TODO mixed precision is not tested
    if melanomamodel.opt['training']['mixed_precision']:
        with amp.autocast():
            outputs = melanomamodel.model(images)
            loss = melanomamodel.criterion(outputs, labels)
        melanomamodel.scaler.scale(loss).backward()
        melanomamodel.scaler.step(melanomamodel.optimizer)
        melanomamodel.scaler.update()
    else:
        outputs = melanomamodel.model(images)
        loss = melanomamodel.criterion(outputs,
                                       labels.float())  # Need to squeeze [BS, 1] to [BS] and BCE uses float
        loss.backward()
        melanomamodel.optimizer.step()
    return loss

def validate(melanomamodel, val_loader, epoch=1):
    melanomamodel.model = melanomamodel.model.to(melanomamodel.device)
    melanomamodel.model.eval()
    total_loss = 0

    with torch.no_grad():
        loop = tqdm(val_loader, desc="[Val]")

        firstitr = True
        for images, labels in loop:
            images, labels = images.to(melanomamodel.device), labels.to(melanomamodel.device)


            outputs = melanomamodel.model(images)
            loss = melanomamodel.criterion(outputs, labels.float())


            total_loss += loss.item()

            if firstitr:
                all_outputs = outputs.cpu()
                all_labels  = labels.cpu()
                firstitr = False
            else:
                all_outputs = torch.cat((all_outputs, outputs.cpu()), dim=0)
                all_labels  = torch.cat((all_labels, labels.cpu()), dim=0)

        avg_loss = total_loss / len(val_loader)
        metrics = evaluate_metrics(melanomamodel.opt, all_outputs, all_labels, epoch+1)
        log_results(melanomamodel.opt, metrics)
    return avg_loss, metrics

def predict(melanomamodel, val_loader, epoch=1):
    melanomamodel.model = melanomamodel.model.to(melanomamodel.device)
    melanomamodel.model.eval()

    with torch.no_grad():
        loop = tqdm(val_loader, desc="[Val]")

        firstitr = True
        for images, labels in loop:
            images, labels = images.to(melanomamodel.device), labels.to(melanomamodel.device)

            outputs = melanomamodel.model(images)

            if firstitr:
                all_outputs = outputs.cpu()
                firstitr = False
            else:
                all_outputs = torch.cat((all_outputs, outputs.cpu()), dim=0)

    write_kaggle_csv(melanomamodel.opt, val_loader.dataset.files, all_outputs.squeeze(1))



def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt", type=str, default="default.yml", help="the option file")
    parser.add_argument("-s", "--savedmodel", type=str, required=False, help="the model file to test")
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

def main():
    opt = argument_parser()

    if opt['dataset']['savedmodel']:
        print(opt['dataset']['savedmodel'])
        print(opt['dataset']['dataset_test_csv'])

        melanomamodel = MelanomaTest(opt)

        # predict only? This happens when our test file does not have labels
        if melanomamodel.predictmode:
            print("PREDICT MODEL MODE")
            predict(melanomamodel, melanomamodel.val_loader, 1)
        else:
            print("TEST MODEL MODE")
            validate(melanomamodel, melanomamodel.val_loader, 1)
    else:
        print("TRAIN MODEL MODE")
        melanomamodel = MelanomaTrainer(opt)
        train(melanomamodel)

if __name__ == "__main__":
    main()
