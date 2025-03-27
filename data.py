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
import random


def vertical_half_mix(img1, img2):
    """Mix two images vertically: top half from img1, bottom half from img2."""
    w, h = img1.size
    mixed = Image.new("RGB", (w, h))
    top_half = img1.crop((0, 0, w, h // 2))
    bottom_half = img2.crop((0, h // 2, w, h))
    mixed.paste(top_half, (0, 0))
    mixed.paste(bottom_half, (0, h // 2))
    return mixed
class MelanomaDataset(Dataset):
    def __init__(self, opt, mode, root, files, classes, transforms=None, subset=0.5):
        """
        subset: Fraction of the dataset to use (e.g., 0.2 for 20%)
        """
        self.opt = opt
        self.mode = mode
        self.root = root
        
        # Optionally use only a fraction of the files and classes
        if subset < 1.0:
            num_samples = int(len(files) * subset)
            self.files = files[:num_samples]
            self.classes = classes[:num_samples]
        else:
            self.files = files
            self.classes = classes

        # Build transforms (could be conditional based on self.mode)
        self.transforms = self.build_transforms() if transforms is None else transforms

        # Get mixing augmentation options from config
        aug = self.opt['dataset'].get('augmentations', {})
        self.mix_enabled = aug.get('image_mix_enabled', False)
        self.mix_prob = aug.get('image_mix_prob', 1.0)  # default to 100% if enabled

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.root, self.files[item])).convert("RGB")
        class_ = self.classes[item]

        # Optionally apply the mixed augmentation on training data only
        if self.mode == "train" and self.mix_enabled:
            if random.random() < self.mix_prob:
                # Select a random image to mix with
                rand_index = random.randint(0, len(self.files) - 1)
                image2 = Image.open(os.path.join(self.root, self.files[rand_index])).convert("RGB")
                # Apply the mixing augmentation (vertical half mix)
                image = vertical_half_mix(image, image2)

        # Apply the defined transformations (resize, flips, etc.)
        if self.transforms:
            image = self.transforms(image)
        return image, class_

    def __len__(self):
        return len(self.files)


    def build_transforms(self):
        # Always start with resizing based on the desired image size
        base_transforms = [transforms.Resize(self.opt['dataset']['image_size'])]
        
        # Build the training augmentation pipeline
        if self.mode == "train":
            aug = self.opt['dataset'].get('augmentations', {})
            # Add horizontal flip if enabled (p > 0)
            if aug.get('horizontal_flip', 0) > 0:
                base_transforms.append(transforms.RandomHorizontalFlip(p=aug['horizontal_flip']))
            # Add vertical flip if enabled
            if aug.get('vertical_flip', 0) > 0:
                base_transforms.append(transforms.RandomVerticalFlip(p=aug['vertical_flip']))
            # Add random rotation if specified (degrees > 0)
            if aug.get('random_rotation', 0) > 0:
                base_transforms.append(transforms.RandomRotation(degrees=aug['random_rotation']))
            # Add color jitter if specified (non-zero value for brightness/contrast/saturation)
            if aug.get('color_jitter', 0) > 0:
                cj_value = aug['color_jitter']
                base_transforms.append(transforms.ColorJitter(
                    brightness=cj_value,
                    contrast=cj_value,
                    saturation=cj_value
                ))
        
        # Append the common transforms for both train and validation
        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        ])
        
        melanoma_transform = transforms.Compose(base_transforms)
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
