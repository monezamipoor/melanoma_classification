import torch
import torch.nn.functional as F
from tqdm import tqdm
from loss import SVMHingeLoss, SupConLoss
from utils import (
    save_checkpoint, soft_voting_probs_from_logits,
    write_kaggle_csv, log_results, log_test
)
import torch.nn as nn
from metrics import evaluate_metrics
from wandb_helper import wandb_watch, wandb_val_log, wandb_test_log


class ContrastiveSVM:
    def __init__(self, opt, device):
        self.opt = opt
        self.device = device
        # Preinitialize losses
        self.svm_loss = SVMHingeLoss().to(device)
        self.supcon_loss = SupConLoss(
            temperature=opt['model']['contrastive_temperature'],
            margin_threshold=opt['model']['contrastive_margin']
        ).to(device)
        self.temp = opt['model'].get('contrastive_temperature', 0.07)

    def train_batch(self, trainer, images, labels, epoch):
        trainer.optimizer.zero_grad()
        x1, x2 = images
        x1, x2, labs = x1.to(self.device), x2.to(self.device), labels.to(self.device)
        f1 = trainer.model(x1, return_features=True)
        f2 = trainer.model(x2, return_features=True)

        if trainer.opt['model'].get('use_contrastive_svm', False):
            o1 = trainer.model.svm_head(f1).squeeze()
            o2 = trainer.model.svm_head(f2).squeeze()
            out = torch.cat([o1, o2], 0);
            lab = torch.cat([labs, labs], 0)
            loss = self.svm_loss(out, lab)
        else:
            p1 = F.normalize(trainer.model.projector(f1), dim=1)
            p2 = F.normalize(trainer.model.projector(f2), dim=1)
            emb = torch.cat([p1, p2], 0)
            lab = torch.cat([labs, labs], 0)
            sim = emb @ emb.T / self.temp
            mask = (lab.unsqueeze(0) == lab.unsqueeze(1)).float()
            logits_mask = torch.eye(mask.size(0), device=mask.device)
            exp_sim = torch.exp(sim) * (1 - logits_mask)
            log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-9)
            pos_per_row = mask.sum(1) + 1e-9
            mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_row
            loss = - mean_log_prob_pos.mean()

        loss.backward()
        trainer.optimizer.step()
        return loss

    def validate(self, trainer, val_loader, epoch):
        in_cl = trainer.opt['model']['loss_function']=='contrastive' and epoch<trainer.opt['training']['contrastive_epochs']
        outputs, labels = [], []
        total = 0.0
        mode = "[Val: Contrastive]" if in_cl else "[Val]"
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=mode):
                (x1, x2), y = batch
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                if trainer.opt['model'].get('use_contrastive_svm', False):
                    o1 = trainer.model.svm_head(trainer.model(x1, return_features=True)).squeeze()
                    o2 = trainer.model.svm_head(trainer.model(x2, return_features=True)).squeeze()
                    out = torch.cat([o1, o2], 0)
                    lab = torch.cat([y, y], 0)
                    loss = self.svm_loss(out, lab)
                else:
                    p1 = F.normalize(trainer.model.projector(trainer.model(x1, return_features=True)),1)
                    p2 = F.normalize(trainer.model.projector(trainer.model(x2, return_features=True)),1)
                    out = torch.cat([p1, p2],0)
                    lab = torch.cat([y, y],0)
                    loss = self.supcon_loss(out, lab)
                total += loss.item(); outputs.append(out.cpu()); labels.append(lab.cpu())

        avg = total/len(val_loader)
        if in_cl:
            print(f"⚡ Contrastive val @ epoch {epoch}, loss={avg:.4f}")
            return avg, {}
        outs = torch.cat(outputs)
        labs = torch.cat(labels)
        mets = evaluate_metrics(trainer.opt, outs, labs, epoch+1)
        log_results(trainer.opt, mets)
        return avg, mets

    def validate_loss(self, trainer, acc_loss, val_loader, desc="[Val]"):
        outs, labs = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=desc):
                images, labels = batch
                labels = labels.to(self.device)
                if isinstance(images, (list,tuple)):
                    x1, x2 = images; x1,x2 = x1.to(self.device), x2.to(self.device)
                    f1 = trainer.model(x1, return_features=True)
                    f2 = trainer.model(x2, return_features=True)
                    if trainer.opt['model'].get('use_contrastive_svm', False):
                        o1 = trainer.model.svm_head(f1).squeeze(); o2 = trainer.model.svm_head(f2).squeeze()
                        out = torch.cat([o1,o2],0); lab = torch.cat([labels,labels],0)
                    else:
                        p1=F.normalize(trainer.model.projector(f1),1); p2=F.normalize(trainer.model.projector(f2),1)
                        out=torch.cat([p1,p2],0); lab=torch.cat([labels,labels],0)
                    loss = (trainer.criterion if trainer.opt['model']['loss_function']=='contrastive' else trainer.criterion_second)(out, lab.float())
                else:
                    imgs = images.to(self.device); out=trainer.model(imgs); loss=trainer.criterion(out, labels.float()); lab=labels
                acc_loss += loss.item(); outs.append(out.cpu()); labs.append(lab.cpu())
        return torch.cat(labs), torch.cat(outs), acc_loss

    def test(self, opt, model_list, val_loader, tag="natural"):
        if not model_list:
            print("Test: no models."); return
        pm = model_list[0].predictmode; print(f"Test predict only={pm}")
        outs, ground = [], None
        for mtest in model_list:
            print(f"Model {mtest.model_path}")
            mtest.model.eval()
            lbls, out, _ = self.validate_loss(mtest, 0.0, val_loader, desc='[Test]')
            ground = ground or lbls
            if out.dim()>1 and out.size(1)==1: out=out.squeeze(1)
            if ground.dim()>1 and ground.size(1)==1: ground=ground.squeeze(1)
            outs.append(out)
        logits = torch.stack(outs,0); probs=soft_voting_probs_from_logits(logits)
        if pm:
            write_kaggle_csv(opt, val_loader.dataset.files, probs, tag=tag)
        else:
            mets=evaluate_metrics(opt, probs, ground, epoch="Test"); log_test(opt,mets,tag=tag); wandb_test_log(mets,tag=tag)

    def on_contrastive_phase_end(self, trainer, epoch):

        ce = trainer.opt['training']['contrastive_epochs']
        if trainer.opt['model']['loss_function'] == 'contrastive' and epoch + 1 == ce:
            print("🧪 Removing contrastive components before saving...")
            # remove projector & contrastive heads
            trainer.model.projector = None
            trainer.model.use_contrastive_head = False
            trainer.model.use_svm_head = False
            trainer.model.training_phase = 'finetune'

            # rebuild final classifier
            feature_dim = (
                trainer.model.backbone.num_features
                if hasattr(trainer.model.backbone, 'num_features')
                else 1280
            )
            trainer.model.classifier = nn.Sequential(
                nn.Dropout(trainer.opt['model']['dropout_rate']),
                nn.Linear(feature_dim, 1)
            ).to(trainer.device)

            # switch loss to the second (BCE) criterion
            trainer.criterion = trainer.criterion_second
            trainer.opt['model']['loss_function'] = 'bce'
