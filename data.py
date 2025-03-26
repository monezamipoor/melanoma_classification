import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time

class MelanomaDataset(Dataset):
    def __init__(self, opt, mode, root, files, classes, transforms=None):

        # Set opt locally
        self.opt = opt
        # Are we train, val or test?
        self.mode = mode
        # location of the dataset
        self.root = root
        # list of files
        self.files = files
        # list of classes
        self.classes = classes
        # transforms
        self.transforms = self.build_transforms()              #TODO Transformations

    def __getitem__(self, item):
        # read the image
        image = Image.open(os.path.join(self.root, self.files[item])).convert(mode="RGB")
        # class for that image
        class_ = self.classes[item]
        # apply transformation
        if self.transforms:
            image = self.transforms(image)

        # return the image and class
        return image, class_

    def __len__(self):
        # return the total number of images
        return len(self.files)

    def build_transforms(self):
        if self.mode == "train":
            melanoma_transform = transforms.Compose([
                transforms.Resize(self.opt['dataset']['image_size']),
                # Add your augmentations here based on opt
                # Example:
                # if opt['dataset']['augmentations']['horizontal_flip'] > 0:
                #     transforms.RandomHorizontalFlip(p=opt['dataset']['augmentations']['horizontal_flip']),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        if self.mode == "val":
            melanoma_transform = transforms.Compose([
                transforms.Resize(self.opt['dataset']['image_size']),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
        return melanoma_transform

# TODO this needs implementing properly and testing?
def get_sampler(dataset, oversampling_rate=1.0, use_stratified=False):
    # If use_stratified, use some positive samples in each batch
    # If oversampling, increase weights for minority classes
    # Create and return sampler uisng params
    return WeightedRandomSampler()

def melanoma_dataloaders(opt):

    dataset = pd.read_csv(opt['dataset']['dataset_train_csv'])

    files = dataset['image_name'].values + '.jpg'       # Images need .jpg to be found
    classes = dataset['target'].values                  # Target classes (0 = benign, 1 = malignant)

    # Split the dataset into 80/20 and default stratify along classes for the split. Note this does not stratify based on batches
    train_files, val_files, train_classes, val_classes = train_test_split(files, classes, train_size=0.8,
                                                                            test_size=0.2, stratify=classes)

    #opt, root, files, classes, transforms=None
    train_dataset = MelanomaDataset(
        opt,
        'train',
        opt['dataset']['dataset_train_path'], train_files, train_classes,
        transforms=None             #TODO Could drop this if transforms are being handled via config.
    )
    
    val_dataset = MelanomaDataset(
        opt,
        'val',
        opt['dataset']['dataset_val_path'], val_files, val_classes,
        transforms=None             #TODO Could drop this if transforms are being handled via config.
    )

    train_sampler = None

    #TODO Configure sampler, straified batching and k-fold.
    '''
    if opt['dataset'].get('oversampling_rate', 1.0) > 1.0 or opt['dataset'].get('stratified_batching', False):
        train_sampler = get_sampler(
            train_dataset,
            oversampling_rate=opt['dataset'].get('oversampling_rate', 1.0),
            use_stratified=opt['dataset'].get('stratified_batching', False)
        )
    '''
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=opt['dataset']['batch_size'],
        sampler=train_sampler, 
        shuffle=True,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=opt['dataset']['batch_size'],
        shuffle=False,
        num_workers=4
    )
    
    return train_loader, val_loader