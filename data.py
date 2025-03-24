import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np

class MelanomaDataset(Dataset):
    def __init__(self, opt, data_path, mode='train'):
        self.opt = opt
        self.data_path = data_path
        self.mode = mode
        self.transforms = self.build_transforms()
        
    def __len__(self):
        # Return the size of the dataset
        pass
        
    def __getitem__(self, idx):
        # Return a sample from the dataset
        pass

    def build_transforms(self):
        if self.mode == "train":
            transforms = transforms.Compose([
                transforms.Resize(self.opt['dataset']['image_size']),
                # Add your augmentations here based on opt
                # Example:
                # if opt['dataset']['augmentations']['horizontal_flip'] > 0:
                #     transforms.RandomHorizontalFlip(p=opt['dataset']['augmentations']['horizontal_flip']),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        
        if self.mode == "val":
            transforms = transforms.Compose([
                transforms.Resize(self.opt['dataset']['image_size']),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
        return transforms

def get_sampler(dataset, oversampling_rate=1.0, use_stratified=False):
    # If use_stratified, use some positive samples in each batch
    # If oversampling, increase weights for minority classes
    # Create and return sampler uisng params
    return WeightedRandomSampler()

def melanoma_dataloaders(opt):
    
    train_dataset = MelanomaDataset(
        opt,
        mode='train',
        data_path=opt['dataset']['dataset_train_path'],
    )
    
    val_dataset = MelanomaDataset(
        opt,
        mode='val',
        data_path=opt['dataset']['dataset_val_path'],
    )

    train_sampler = None
    if opt['dataset'].get('oversampling_rate', 1.0) > 1.0 or opt['dataset'].get('stratified_batching', False):
        train_sampler = get_sampler(
            train_dataset,
            oversampling_rate=opt['dataset'].get('oversampling_rate', 1.0),
            use_stratified=opt['dataset'].get('stratified_batching', False)
        )
    
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