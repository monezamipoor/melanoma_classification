import argparse
import yaml
import os
import torch
import torch.cuda.amp as amp
from torch import optim
from tqdm import tqdm

from data import melanoma_train_dataloaders, melanoma_test_dataloaders
from model import train_melanoma_model, test_melanoma_model
from model_hybrid import train_hybrid_model, test_hybrid_model
from loss import melanoma_loss
from utils import log_results, cuda_available, log_model, save_checkpoint, write_kaggle_csv, \
    soft_voting_probs_from_logits, log_test
from metrics import evaluate_metrics
from wandb_helper import wandb_login, wandb_watch, wandb_train_log, wandb_val_log, wandb_test_log


#Uncomment to turn off wandb entirely for debugging only
#wandb.init(mode="disabled")

# Refactored the denorm image and save augmented samples method to be in utils.py

class MelanomaTest:
    def __init__(self, opt, testmodel):
        self.opt = opt
        print(opt)
        self.device = cuda_available(self.opt)

        self.predictmode, self.val_loader = melanoma_test_dataloaders(opt)
        self.is_kfold = False

        if opt['model']['hybrid'].get('enabled', False):            # Use the hybrid model config
            self.model = test_hybrid_model(opt, testmodel).to(self.device)
            print("Using Hybrid Model")
        else:                                                       # Use the basic model config
            self.model = test_melanoma_model(opt, testmodel).to(self.device)

        self.model_path = testmodel
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
        for fold_data in melanomamodel.fold_loaders:
            fold_idx = fold_data['fold']
            print(f"\n[INFO] Starting Fold {fold_idx}")

            # Re-initialize the model for each fold (bassically a fresh start):
            melanomamodel.model = train_melanoma_model(melanomamodel.opt).to(melanomamodel.device)
            melanomamodel.optimizer = melanomamodel.get_optimizer()
            melanomamodel.scheduler = melanomamodel.get_scheduler()
            melanomamodel.best_metrics = melanomamodel.reset_metrics()      # Fixed for folds using past folds metrics for comparison.

            train_loader = fold_data['train_loader']
            val_loader   = fold_data['val_loader']

            wandb_watch(melanomamodel.model, melanomamodel.criterion, log_freq=10)

            modeltokeep = None

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
            melanomamodel.model = melanomamodel.model.to(melanomamodel.device)
            melanomamodel.model.train()
            total_loss = 0

            loop = tqdm(melanomamodel.train_loader, desc=f"Epoch {epoch + 1}/{melanomamodel.opt['training']['epochs']}")

            #If you want to see the images after Aug
            # save_augmented_samples(train_loader_p, num_samples=10, save_dir="/content/drive/MyDrive/melanoma_classification/logs/Sample")

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

            savedmodel = save_checkpoint(melanomamodel.opt, melanomamodel.best_metrics, melanomamodel.model, epoch + 1, val_metrics)
            if savedmodel is not None:
                testmodels = [savedmodel]

    return testmodels


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

    # Refactored the loss loop to "validate_loss"
    with torch.no_grad():
        all_labels, all_outputs, total_loss = validate_loss(melanomamodel, total_loss, val_loader)
        avg_loss = total_loss / len(val_loader)

        # CONVERT OUTPUTS TO PROBS FOR METRICS
        probabilities = torch.sigmoid(all_outputs)

        metrics = evaluate_metrics(melanomamodel.opt, probabilities, all_labels, epoch+1)
        log_results(melanomamodel.opt, metrics)

        print_raw_logits_and_probs(all_labels, all_outputs)

    return avg_loss, metrics

# validation loss calc refactored out to be called from both "validate" and "test"
def validate_loss(melanomamodel, total_loss, val_loader, description="[Val]"):
    loop = tqdm(val_loader, desc=description)
    firstitr = True
    for images, labels in loop:
        images, labels = images.to(melanomamodel.device), labels.to(melanomamodel.device)

        outputs = melanomamodel.model(images)
        loss = melanomamodel.criterion(outputs, labels.float())

        total_loss += loss.item()

        if firstitr:
            all_outputs = outputs.cpu()
            all_labels = labels.cpu()
            firstitr = False
        else:
            all_outputs = torch.cat((all_outputs, outputs.cpu()), dim=0)
            all_labels = torch.cat((all_labels, labels.cpu()), dim=0)
    return all_labels, all_outputs, total_loss

# For post training / validation tests of saved models, or from command line.
def test(opt, melanoma_model_list, val_loader):

    if melanoma_model_list is None or len(melanoma_model_list) == 0:
        print("Test: No models to test. Exiting...")
        return

    predictonly = melanoma_model_list[0].predictmode        # True = we don't have class labels. False = we do and can mark our own homework
    print("Test: Generate predictions only = {}".format(predictonly))

    output_list = []

    # Loop each model we want to generate predictions for. This supports both single models and cross-validation
    for melanoma_test in melanoma_model_list:
        print("Test: Model {}".format(melanoma_test.model_path))
        melanoma_test.model = melanoma_test.model.to(melanoma_test.device)
        melanoma_test.model.eval()

        total_loss = 0
        with torch.no_grad():
            # TODO. For predictonly=True operations we don't need to calc loss or capture labels (they are all -1). But currently there is no detrimental effect reusing this call for simplicity/consistency of operation.
            all_labels, all_outputs, total_loss = validate_loss(melanoma_test, total_loss, val_loader, description='[Test]')

        # Safety check for tensor dimension
        if all_outputs.dim() > 1 and all_outputs.shape[1] == 1:
            all_outputs = all_outputs.squeeze(1)
        # Safety check for tensor dimension
        if all_labels.dim() > 1 and all_labels.shape[1] == 1:
            all_labels = all_labels.squeeze(1)
        output_list.append(all_outputs)

    ensemble_logits = torch.stack(output_list, dim=0)
    probabilities = soft_voting_probs_from_logits(ensemble_logits)  # Functions identically for non k-fold tests as it simply applies a mean to a dimension of 1. (i.e. no change)

    # Only writing a kaggle csv if we have no labels
    if predictonly:
        write_kaggle_csv(opt, val_loader.dataset.files, probabilities)
    else:
        metrics = evaluate_metrics(opt, probabilities, all_labels, epoch='Test')
        log_test(opt, metrics)
        wandb_test_log(**metrics)
        print(f"Test Metrics: {metrics}")

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

def main():
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

    # When calling our test method we force all models to use the same data loader (nominally from the first one)
    test(opt, melanomatests, melanomatests[0].val_loader)

if __name__ == "__main__":
    main()
