"""
This module contains the ContrastiveSVM class, which implements the contrastive idea for Pretraining the model.
It incluses the training and validation methods for the contrastive loss. 
The supcon loss is behind the contrastive code and it located in the loss.py file.
The class works with two pairs of images, and it is used to train the model with a contrastive loss.
The main idea is to make closer the features of the same class and to make farther the features of different classes.

"""


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
    """
    ContrastiveSVM class for training and validating a model with contrastive loss.
    This class is used to train the model with a contrastive loss, which is a type of loss function that
    encourages the model to learn representations that are close together for similar inputs and far apart for dissimilar inputs.

    """
    def __init__(self, opt, device):
        """
        Initialize the ContrastiveSVM class.
        
        """
        # Initialize the ContrastiveSVM class.
        self.opt = opt
        # Set the device for training. it can be either 'cuda' or 'cpu'.
        self.device = device
        # Preinitialize losses. it is deactivated in the constructor and no more well be used.
        self.svm_loss = SVMHingeLoss().to(device)
        # If the model is using contrastive loss, initialize the contrastive loss function.
        self.supcon_loss = SupConLoss(
            temperature=opt['model']['contrastive_temperature'],
            margin_threshold=opt['model']['contrastive_margin']
        ).to(device)
        #it is used to set the temperature for the contrastive loss.
        # Set the temperature for the contrastive loss. it is used to scale the similarity scores. 
        self.temp = opt['model'].get('contrastive_temperature', 0.07)

    def train_batch(self, trainer, images, labels, epoch):
        """
        Train a batch of images with contrastive loss.
        
        """
        # call the optimizer to zero the gradients.
        trainer.optimizer.zero_grad()
        # take a pairs of images.
        x1, x2 = images
        x1, x2 = x1.to(self.device), x2.to(self.device)
        # take the labels and move them to the device.
        labs = labels.to(self.device)

        # We need to return features to projector.
        f1 = trainer.model(x1, return_features=True)
        f2 = trainer.model(x2, return_features=True)

        # If the model is using contrastive SVM, use the SVM head to get the output. 
        # Otherwise, use the projector to get the output.
        # The SVM head is used to get the output for the SVM loss.
        # Note: the SVM approach is not used any more.
        # The projector is used to get the output for the contrastive loss.
        if trainer.opt['model'].get('use_contrastive_svm', False):
            # If the model is using contrastive SVM, use the SVM head to get the output.
            o1 = trainer.model.svm_head(f1).squeeze()
            # Using the SVM head instead of normal head.
            o2 = trainer.model.svm_head(f2).squeeze()
            # Concatenate the outputs of the two images.
            out = torch.cat([o1, o2], dim=0)
            # Concatenate the labels of the two images.
            lab = torch.cat([labs, labs], dim=0)
            # Calculate the loss using the SVM loss function.
            loss = self.svm_loss(out, lab)
        else:
            # If the model is not using contrastive SVM, use the projector to get the output.
            # Normalize the output of the projector.
            p1 = F.normalize(trainer.model.projector(f1), dim=1)
            # Normalize the output of the projector.
            # Normalize the output of the projector.
            p2 = F.normalize(trainer.model.projector(f2), dim=1)
            # Concatenate the outputs of the two images.
            emb = torch.cat([p1, p2], dim=0)
            # Concatenate the labels of the two images.
            lab = torch.cat([labs, labs], dim=0)
            # Calculate the loss using the contrastive loss function.
            # The contrastive loss is used to learn representations that are close together for similar inputs and far apart for dissimilar inputs.
            loss = self.supcon_loss(emb, lab)  

        # Calculate the gradients of the loss with respect to the model parameters.
        # The gradients are used to update the model parameters.
        loss.backward()
        # Update the model parameters using the optimizer.
        trainer.optimizer.step()
        return loss


    def validate(self, trainer, val_loader, epoch):
        """
        Validate the model on the validation set.
        This function is used to evaluate the model on the validation set and calculate the loss and metrics.
        It is used to evaluate the model on the validation set and calculate the loss and metrics.
        The function takes the trainer object, the validation loader, and the current epoch as input.
        The function returns the average loss and the metrics.

        """
        # Check if the model is using contrastive loss and if the current epoch is less than the contrastive epochs. We set the contrastive epochs in the config file.
        # If the model is using contrastive loss, we need to use the contrastive loss function.
        in_cl = trainer.opt['model']['loss_function']=='contrastive' and epoch<trainer.opt['training']['contrastive_epochs']
        # Create empty lists to store the outputs and labels.
        # The outputs are the predictions of the model and the labels are the ground truth labels.
        outputs, labels = [], []
        total = 0.0
        # Split the validation set into two sections: one for the contrastive loss and one for the normal loss.
        mode = "[Val: Contrastive]" if in_cl else "[Val]"
        # Set the model to evaluation mode by using no_grad to disable gradient calculation.
        with torch.no_grad():
            # Iterate over the validation set.
            for batch in tqdm(val_loader, desc=mode):
                # Use two representations of the same image.
                # The first representation is the original image and the second representation is the augmented image.
                (x1, x2), y = batch
                # Move the images and labels to the device.
                # The images are the input to the model and the labels are the ground truth labels.
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                # If the model is using contrastive SVM, use the SVM head to get the output.
                if trainer.opt['model'].get('use_contrastive_svm', False):
                    # If the model is using contrastive SVM, use the SVM head to get the output.
                    # Feeding the image to the model and getting the features. Then we use the SVM head to get the output.
                    o1 = trainer.model.svm_head(trainer.model(x1, return_features=True)).squeeze()
                    # Using the squeeze method to remove the dimension of size 1 from the output. Remove singleton dimensions.
                    o2 = trainer.model.svm_head(trainer.model(x2, return_features=True)).squeeze()
                    # Concatenate the outputs of the two images.
                    out = torch.cat([o1, o2], 0)
                    # Concatenate the labels of the two images.
                    lab = torch.cat([y, y], 0)
                    # Calculate the loss using the SVM loss function.
                    loss = self.svm_loss(out, lab)
                else:
                    # If the model is not using contrastive SVM, use the projector to get the output.
                    # Feeding the image to the model and getting the features. Then we use the projector to get the output.
                    p1 = F.normalize(trainer.model.projector(trainer.model(x1, return_features=True)),1)
                    # Normalize the output of the projector.
                    p2 = F.normalize(trainer.model.projector(trainer.model(x2, return_features=True)),1)
                    # Normalize the output of the projector.
                    # Concatenate the outputs of the two images.
                    out = torch.cat([p1, p2],0)
                    # Concatenate the labels of the two images.
                    # The labels are the ground truth labels.
                    lab = torch.cat([y, y],0)
                    # Calculate the loss using the contrastive loss function.
                    # The contrastive loss is used to learn representations that are close together for similar inputs and far apart for dissimilar inputs.
                    # if the model is using contrastive loss, we need to use the contrastive loss function.
                    loss = self.supcon_loss(out, lab)
                # If the model is not using contrastive SVM, use the projector to get the output.
                # Calculate the loss using the contrastive loss function.   
                total += loss.item(); outputs.append(out.cpu()); labels.append(lab.cpu())
        # Calculate the average loss by dividing the total loss by the number of batches.
        # The average loss is used to evaluate the model on the validation set.
        avg = total/len(val_loader)
        # If the model is using contrastive loss, we need to use the contrastive loss function.
        if in_cl:
            print(f" Contrastive val @ epoch {epoch}, loss={avg:.4f}")
            # print logs for just contrastive loss.
            return avg, {}
        # aggregate the outputs and labels from all batches.
        outs = torch.cat(outputs)
        labs = torch.cat(labels)
        # We are not using classification methosd, so we need to use the sigmoid method to get the probabilities and in this stage is not neccesary to have metrics.We just need the contrastive for pretraining.
        mets = evaluate_metrics(trainer.opt, outs, labs, epoch+1)
        # Log the results of the validation.
        log_results(trainer.opt, mets)
        return avg, mets

    def validate_loss(self, trainer, acc_loss, val_loader, desc="[Val]"):
        """
        Validate the model on the validation set and calculate the loss.
        this function looks at the model regardless of the pretraining phase. It dosen't matter the model used as pretrained or had a contrastive phase.
        It is used to evaluate the model on the validation set and calculate the loss and metrics.

        """
        # define two empty lists to store the outputs and labels.
        outs, labs = [], []
        # Set the model to evaluation mode by using no_grad to disable gradient calculation.
        # The model is in evaluation mode, so it will not update the weights.
        with torch.no_grad():
            # Iterate over the validation set.
            for batch in tqdm(val_loader, desc=desc):
                # extract the images and labels from the batch.
                images, labels = batch
                # Move the images and labels to the device.
                labels = labels.to(self.device)
                # check if the image is a pair of images or a single image. Usually, it happens when we have contrastive method.
                if isinstance(images, (list,tuple)):
                    # If the images are a pair of images, we need to use the projector to get the output.
                    x1, x2 = images; x1,x2 = x1.to(self.device), x2.to(self.device)
                    # We need to return features to projector.
                    # Feeding the image to the model and getting the features. Then we use the projector to get the output.
                    f1 = trainer.model(x1, return_features=True)
                    # Normalize the output of the projector.
                    f2 = trainer.model(x2, return_features=True)
                    # Normalize the output of the projector.
                    # If the model is using contrastive SVM, use the SVM head to get the output.
                    if trainer.opt['model'].get('use_contrastive_svm', False):
                        # having the output of the SVM head after removing the dimension of size 1 from the output.
                        o1 = trainer.model.svm_head(f1).squeeze(); o2 = trainer.model.svm_head(f2).squeeze()
                        # Using the squeeze method to remove the dimension of size 1 from the output. Remove singleton dimensions.
                        # Concatenate the outputs of the two images.
                        # Concatenate the labels of the two images.
                        out = torch.cat([o1,o2],0); lab = torch.cat([labels,labels],0)
                    else:
                        # If the model is not using contrastive SVM, use the projector to get the output.
                        # Feeding the image to the model and getting the features. Then we use the projector to get the output.
                        # Normalize the output of the projector.
                        p1=F.normalize(trainer.model.projector(f1),1); p2=F.normalize(trainer.model.projector(f2),1)
                        out=torch.cat([p1,p2],0); lab=torch.cat([labels,labels],0)
                    # Calculate the loss using the contrastive loss function.
                    # The contrastive loss is used to learn representations that are close together for similar inputs and far apart for dissimilar inputs.
                    loss = (trainer.criterion if trainer.opt['model']['loss_function']=='contrastive' else trainer.criterion_second)(out, lab.float())
                else:
                    # If the images are a single image, we dont need to use the projector to get the output.
                    # Move the images to the device.
                    imgs = images.to(self.device); out=trainer.model(imgs); loss=trainer.criterion(out, labels.float()); lab=labels
                    
                acc_loss += loss.item(); outs.append(out.cpu()); labs.append(lab.cpu())
        # Calculate the average loss by dividing the total loss by the number of batches.
        return torch.cat(labs), torch.cat(outs), acc_loss

    def test(self, opt, model_list, val_loader, tag="natural"):
        """
        Test the model on the test set and calculate the metrics. We are not using the contrastive loss in this stage.
        This function is just using the normal classification method to evaluate the model.
        
        """
        # Check if the model list is empty. If it is empty, we need to return.
        if not model_list:
            print("Test: no models."); return
        # if the predict mode is set to True, we need to use the predict mode.
        pm = model_list[0].predictmode; print(f"Test predict only={pm}")
        # define two empty lists to store the outputs and labels.
        outs, ground = [], None
        # iterate over the model list that saved in the checkpoint.
        # The model list is a list of models that are used to evaluate the model and we want to use the best one based on the AUC score.
        for mtest in model_list:
            # Load the model from the checkpoint.
            print(f"Model {mtest.model_path}")
            # Set the model to evaluation mode.
            mtest.model.eval()
            # Set the model to the device.
            # Take the labels and output by using the validate_loss function. We are not useing loss results in this stage.
            lbls, out, _ = self.validate_loss(mtest, 0.0, val_loader, desc='[Test]')

            # if we have groung truth labels, then we use it. Otherwise, we use the labels from the validation set.
            ground = ground or lbls
            # if the dimension of output is greater than 1 and the size of the output in the second dimension is 1, we need to squeeze the output.
            if out.dim()>1 and out.size(1)==1: out=out.squeeze(1)
            # if the dimension of ground is greater than 1 and the size of the ground in the second dimension is 1, we need to squeeze the ground.
            if ground.dim()>1 and ground.size(1)==1: ground=ground.squeeze(1)
            # We update the list by new output.
            outs.append(out)

        # if we have just logits, we need to use the soft voting method to get the probabilities from the outputs.
        # We need to use the soft voting method to get the probabilities from the outputs.
        # The soft voting method is used to get the probabilities from the outputs.
        logits = torch.stack(outs,0); probs=soft_voting_probs_from_logits(logits)
        # if the model is using the predict mode, we need to use the predict mode.
        if pm:
            # if we want to provide the result for the competition, we need to use the kaggle version.
            write_kaggle_csv(opt, val_loader.dataset.files, probs, tag=tag)
        else:
            # Otherwise, we need to use the normal version.
            mets=evaluate_metrics(opt, probs, ground, epoch="Test"); log_test(opt,mets,tag=tag); wandb_test_log(mets,tag=tag)

    def on_contrastive_phase_end(self, trainer, epoch):
        """
        This function is used to change the config file for transition from contrastive to normal training.
        In this section we remove the contrastive components and we switch to the second loss function.
        The function takes the trainer object and the current epoch as input.

        """
        # Check if the contrastive phase is ended.
        ce = trainer.opt['training']['contrastive_epochs']
        # it is a transition from contrastive to normal training.
        if trainer.opt['model']['loss_function'] == 'contrastive' and epoch + 1 == ce:
            print(" Removing contrastive components before saving...")
            # remove projector & contrastive heads
            trainer.model.projector = None
            trainer.model.use_contrastive_head = False
            trainer.model.use_svm_head = False
            # Switch to Finetune phase as a normal training.
            trainer.model.training_phase = 'finetune'

            # rebuild final classifier
            feature_dim = (
                trainer.model.backbone.num_features
                if hasattr(trainer.model.backbone, 'num_features')
                else 1280
            )
            # Replace the projector with a new classifier.
            # The projector is used to get the output for the contrastive loss.
            trainer.model.classifier = nn.Sequential(
                nn.Dropout(trainer.opt['model']['dropout_rate']),
                nn.Linear(feature_dim, 1)
            ).to(trainer.device)

            # switch loss to the second (BCE) criterion
            trainer.criterion = trainer.criterion_second
            trainer.opt['model']['loss_function'] = 'bce'
