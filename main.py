import wandb
import argparse
import yaml
import os
import torch
import torch.cuda.amp as amp
from timm.utils import AverageMeter
from torch import nn, optim
from tqdm import tqdm
from data import melanoma_dataloaders
from model import melanoma_model, melanoma_loss
from utils import log_results, cuda_available
from metrics import evaluate_metrics
from datetime import datetime
import torch.nn.functional as F

class MelanomaTrainer:
    def __init__(self, opt):
        self.opt = opt
        print(opt)
        self.device = cuda_available(self.opt)
        self.train_loader, self.val_loader = melanoma_dataloaders(opt)
        self.model = melanoma_model(opt).to(self.device)
        self.criterion = melanoma_loss(opt)
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_scheduler()
        self.scaler = amp.GradScaler() if opt['training']['mixed_precision'] else None
        self.best_metrics = {metric: float('-inf') for metric in opt['testing']['model_save_metric']}

        if opt['training']['freeze_pretrained']:
            self.freeze_backbone(bool(opt['training']['freeze_pretrained']))
        else:
            self.freeze_backbone(False)

        if opt['testing']['wandb']['project_name']:
            wandb.init(project=opt['testing']['wandb']['project_name'], entity=opt['testing']['wandb']['entity'])
            wandb.config.update(opt)

    def get_optimizer(self):
        if self.opt['training']['optimizer'] == 'adam':
            return optim.Adam(self.model.parameters(), lr=self.opt['training']['learning_rate'])
        elif self.opt['training']['optimizer'] == 'sgd':
            return optim.SGD(self.model.parameters(), lr=self.opt['training']['learning_rate'], momentum=0.9)
        elif self.opt['training']['optimizer'] == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=self.opt['training']['learning_rate'])

    def get_scheduler(self):
        if self.opt['training']['scheduler'] == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.opt['training']['epochs'])
        elif self.opt['training']['scheduler'] == 'step':
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.opt['training']['step_size'], gamma=self.opt['training']['decay_rate'])
        elif self.opt['training']['scheduler'] == 'reduce_on_plateau':
            return torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5, factor=0.1)

    def freeze_backbone(self, freeze=False):
        for param in list(self.model.parameters())[:-1]:
            param.requires_grad = freeze
        print("Backbone layers frozen.= " + str(freeze))

    def save_checkpoint(self, epoch, metrics):
        save_strategy = self.opt['testing']['model_save_strategy']
        if save_strategy == 'none':
            return
        timestamp = datetime.now().isoformat(timespec='minutes')
        checkpoint_dir = self.opt['testing']['checkpoint_dir']
        os.makedirs(checkpoint_dir, exist_ok=True)

        if save_strategy == 'best':
            for metric in self.opt['testing']['model_save_metric']:
                if metrics[metric] > self.best_metrics[metric]:
                    self.best_metrics[metric] = metrics[metric]
                    save_path = os.path.join(
                        checkpoint_dir,
                        f"{timestamp}_{os.path.basename(self.opt['opt']).replace('.yml', '')}_epoch_{epoch+1}_{metric}.pth"
                    )
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Saved best model for {metric} at epoch {epoch+1}")

        elif save_strategy == 'last':
            save_path = os.path.join(
                checkpoint_dir,
                f"{timestamp}_{os.path.basename(self.opt['opt']).replace('.yml', '')}_epoch_{epoch+1}_last.pth"
            )
            torch.save(self.model.state_dict(), save_path)
            print(f"Saved last model at epoch {epoch+1}")

        elif save_strategy == 'all':
            for metric in self.opt['testing']['model_save_metric']:
                save_path = os.path.join(
                    checkpoint_dir,
                    f"{timestamp}_{os.path.basename(self.opt['opt']).replace('.yml', '')}_epoch_{epoch+1}_{metric}.pth"
                )
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved model for {metric} at epoch {epoch+1}")

    def train(self):
        print("Starting Training")
        for epoch in range(self.opt['training']['epochs']):
            self.model = self.model.to(self.device)
            self.model.train()
            total_loss = 0

            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.opt['training']['epochs']}")

            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()

                #TODO mixed precision is not tested
                if self.opt['training']['mixed_precision']:
                    with amp.autocast():
                        outputs = self.model(images)
                        loss = self.criterion(outputs, labels)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs.squeeze(1), labels.float())       # Need to squeeze [BS, 1] to [BS] and BCE uses float
                    loss.backward()
                    self.optimizer.step()

                if self.opt['training']['gradient_clipping']:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['training']['gradient_clipping'])

                total_loss += loss.item()
                loop.set_postfix(loss=loss.item())

            avg_loss = total_loss / len(self.train_loader)
            val_loss, val_metrics = self.validate()             #TODO Would this be better extracted outside of the train method?
            self.scheduler.step(val_loss if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) else None)

            print(f"Epoch {epoch+1} - Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, Metrics: {val_metrics}")

            if self.opt['testing']['wandb']['project_name']:
                wandb.log({"Train Loss": avg_loss, "Validation Loss": val_loss, **val_metrics})

            #TODO with metrics implementation
            self.save_checkpoint(epoch, val_metrics)

    def validate(self):
        self.model = self.model.to(self.device)
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating")

            firstitr = True

            for images, labels in loop:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                # loss = self.criterion(outputs.squeeze(), labels.float())
                loss = self.criterion(outputs.view(-1), labels.view(-1).float())

                total_loss += loss.item()

                if firstitr:
                    all_outputs = outputs.cpu()
                    all_labels = labels.cpu()
                    firstitr = False
                else:
                    all_outputs = torch.cat((all_outputs, outputs.cpu()), dim=0)
                    all_labels = torch.cat((all_labels, labels.cpu()), dim=0)

        avg_loss = total_loss / len(self.val_loader)
        metrics = evaluate_metrics(self.opt, all_outputs.squeeze(1), all_labels)
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
