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
def stratified_sampler(classes):
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

def up_sampling(files, classes, aug=False, oversampling_rate=2):
    
    files = list(files)
    classes = list(classes)
    
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
