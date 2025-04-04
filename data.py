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

from torchvision.transforms.functional import InterpolationMode


def column_mix(img1, img2, img3, img4):

    w, h = img1.size
    
    mixed = Image.new("RGB", (w, h))

    strip_width = w // 4

    col1 = img1.crop((0, 0, strip_width, h))
    col2 = img2.crop((strip_width, 0, 2 * strip_width, h))
    col3 = img3.crop((2 * strip_width, 0, 3 * strip_width, h))
    col4 = img4.crop((3 * strip_width, 0, w, h))

    mixed.paste(col1, (0, 0))
    mixed.paste(col2, (strip_width, 0))
    mixed.paste(col3, (2 * strip_width, 0))
    mixed.paste(col4, (3 * strip_width, 0))
    
    return mixed

class QuadrantMixTransform:

    def __init__(self, mix_prob, root, files):
        self.mix_prob = mix_prob
        self.root = root
        self.files = files 

    def __call__(self, img):
        # With probability mix_prob, perform quadrant mix.
        if random.random() < self.mix_prob and len(self.files) >= 4:
            # Randomly sample four indices (without replacement).
            indices = random.sample(range(len(self.files)), 4)
            target_size = img.size
            img1 = Image.open(os.path.join(self.root, self.files[indices[0]])).convert("RGB").resize(target_size)
            img2 = Image.open(os.path.join(self.root, self.files[indices[1]])).convert("RGB").resize(target_size)
            img3 = Image.open(os.path.join(self.root, self.files[indices[2]])).convert("RGB").resize(target_size)
            img4 = Image.open(os.path.join(self.root, self.files[indices[3]])).convert("RGB").resize(target_size)
            return column_mix(img1, img2, img3, img4)
        return img

class MelanomaDataset(Dataset):
    def __init__(self, opt, mode, root, files, classes, transforms=None, subset=1.0):
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


    def __getitem__(self, item):
        image = Image.open(os.path.join(self.root, self.files[item])).convert("RGB")
        class_ = self.classes[item]

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
            # Horizontal flip
            if aug.get('horizontal_flip', 0) > 0:
                base_transforms.append(transforms.RandomHorizontalFlip(p=aug['horizontal_flip']))
            # Vertical flip
            if aug.get('vertical_flip', 0) > 0:
                base_transforms.append(transforms.RandomVerticalFlip(p=aug['vertical_flip']))
            # Random rotation
            if aug.get('random_rotation', 0) > 0:
                base_transforms.append(transforms.RandomRotation(
                    degrees=aug['random_rotation'],
                    interpolation=InterpolationMode.NEAREST,
                    fill=(255, 255, 255)  # fill empty areas with white instead of black
                ))
            if aug.get('random_shear', 0) > 0:
                base_transforms.append(transforms.RandomAffine(degrees=0, shear=aug['random_shear'], fill=(255, 255, 255) ))
            if aug.get('shift_vertical', None) is not None:
                vertical_shift = aug['shift_vertical'][1]

                base_transforms.append(transforms.RandomAffine(degrees=0, translate=(0, vertical_shift), fill=(255, 255, 255) ))

            # Color jitter
            if aug.get('color_jitter', 0) > 0:
                cj_value = aug['color_jitter']
                base_transforms.append(transforms.ColorJitter(
                    brightness=cj_value,
                    contrast=cj_value,
                    saturation=cj_value
                ))
            # Add quadrant mixing if enabled
            if aug.get('image_mix_enabled', False):
                mix_prob = aug.get('image_mix_prob', 1.0)
                base_transforms.append(QuadrantMixTransform(mix_prob, self.root, self.files))

        
        base_transforms.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
        return transforms.Compose(base_transforms)
   


# TODO this needs implementing properly and testing?
def stratified_sampler(opt):

    dataset = pd.read_csv(opt['dataset']['dataset_train_csv'])
    classes = list(dataset['target'].values)
    
    # If use_stratified, use some positive samples in each batch
    # If oversampling, increase weights for minority classes
    # Create and return sampler uisng params
    
    classes_arr = np.array(classes)
    
    # Get unique classes and their respective counts
    unique_classes, counts = np.unique(classes_arr, return_counts=True)

    # This will compute the weight for each class by counting the number of samples in each class
    weights_per_class = 1.0 / counts 

    # it will assign a weight to each sample based on its class
    weights = weights_per_class[classes_arr]
    weights = torch.tensor(sample_weights, dtype=torch.float32)
    
    # Create the WeightedRandomSampler. Replacement=True allows oversampling.
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=False
    )
    
    return sampler

def up_sampling(opt, aug=False, oversampling_rate=2):
    
    dataset = pd.read_csv(opt['dataset']['dataset_train_csv'])
    
    files = list(dataset['image_name'].values + '.jpg')
    classes = list(dataset['target'].values)
    
    # Separate class 0 and class 1 samples
    class_0_files = [file_name for file_name, label in zip(files, classes) if label == 0]
    class_1_files = [file_name for file_name, label in zip(files, classes) if label == 1]

    # Option 1 if we want to make the size of the both class exactly like each other 
    # # Get majority count (class 0 count), we know class 1 have 584 and class 0 have 32542 its just for ourself to be completle sure and compare it with after up sampling.
    # majority_count = len(class_0_files)
    # minority_count = len(class_1_files)

    ## Duplicate class 1 files to match class 0 count

    # Option 1 if we want to make the size of the both class exactly like each other 
    # oversampling_class_1 = class_1_files * (majority_count // minority_count) # multiply the amount of diffenernce 
    # remainder = majority_count % minority_count # Because we round the difference it has the possibility that wont the two class be exactly in the same amount so we calculate the remainder 
    # oversampling_class_1 += class_1_files[:remainder] 

    # Option 2 multiply the minority class by an oversampling rate 
    ## Duplicate class 1 files to match class 0 count
    oversampling_class_1 = class_1_files * oversampling_rate 

    # Combine class 0 and oversampled class 1
    new_files = class_0_files + oversampling_class_1
    new_classes = [0] * len(class_0_files) + [1] * len(oversampling_class_1)
    return new_files, new_classes    


def melanoma_dataloaders(opt):

    dataset = pd.read_csv(opt['dataset']['dataset_train_csv'])

    files = dataset['image_name'].values + '.jpg'       # Images need .jpg to be found
    classes = dataset['target'].values                  # Target classes (0 = benign, 1 = malignant)

    if opt['dataset'].get('use_groupkfold', False):

        # Splitting the train and test datasets
        test_dataset = dataset[dataset['tfrecord'].isin([12, 13, 14])]
        train_dataset = dataset[dataset['tfrecord'].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])]

        train_files = train_dataset['image_name'].values + '.jpg'
        train_classes = train_dataset['target'].values
        
        # we will use tfrecord column to grouping the datas and fed it to GroupKFold
        all_groups = train_dataset['tfrecord'].values  
        
        # We will use 3 fold 
        n_splits = opt['dataset'].get('n_splits', 3)  
        GroupKFold = GroupKFold(n_splits=n_splits)
        
        # For simplicity we take the first fold here.
        train_idx, val_idx = next(GroupKFold.split(train_files, train_classes, groups=all_groups))
        
        train_files = train_files[train_idx]
        val_files = train_files[val_idx]
        train_classes = train_classes[train_idx]
        val_classes = train_classes[val_idx]
        
    else:
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
