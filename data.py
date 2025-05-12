'''
data.py - This module handles all data-related operations for the melanoma dataset, including loading, preprocessing, and augmentation.
It defines a custom dataset class that supports: Stratified sampling, Downsampling and upsampling and K-fold cross-validation.
All data pipelines are modular and fully configurable via the project's configuration file.

'''

import os
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import time

import torchvision.transforms.functional as F
from sklearn.model_selection import GroupKFold, train_test_split
from PIL import Image
import numpy as np
import pandas as pd
import random
from torchvision.transforms.functional import InterpolationMode
from torchvision.transforms import RandomErasing
try:
    from torchvision.transforms import ElasticTransform
except ImportError:
    ElasticTransform = None 

import utils

# This function creates a new image by mixing four images in a column-wise manner.
def column_mix(img1, img2, img3, img4):
    """
    Takes four images and creates a new image by mixing them in a column-wise manner.
    The first image is cropped and continously other crroped images are pasted in the next columns in a new image.

     """
    # get the width and height of the first image
    w, h = img1.size
    # create a new image with the same size as the first image
    mixed = Image.new("RGB", (w, h))
    # introduce a strip width to have a exact equation of width for division
    # divide the width of the image by 4 to get the width of each strip
    strip_width = w // 4
    # crop each image into strips and paste them into the new image
    col1 = img1.crop((0, 0, strip_width, h))
    col2 = img2.crop((strip_width, 0, 2 * strip_width, h))
    col3 = img3.crop((2 * strip_width, 0, 3 * strip_width, h))
    col4 = img4.crop((3 * strip_width, 0, w, h))
    # paste the cropped images into the new image
    mixed.paste(col1, (0, 0))
    mixed.paste(col2, (strip_width, 0))
    mixed.paste(col3, (2 * strip_width, 0))
    mixed.paste(col4, (3 * strip_width, 0))
    return mixed

# Creating QuadrantMix Augmentation which creates mosaics of four images in a grid
class QuadrantMixTransform:
    def __init__(self, mix_prob, root, files):
        """ This class read images and pass a set of four images to the column_mix function.
            At the end it returns the mixed image.
        """
        # it is a variable to control the probability of applying the mix
        self.mix_prob = mix_prob
        # root is the path to the images
        self.root = root
        # files is the list of image file names
        self.files = files

    def __call__(self, img):
        # With probability mix_prob, perform quadrant mix.
        if random.random() < self.mix_prob and len(self.files) >= 4:
            # Randomly select 4 images from the dataset
            indices = random.sample(range(len(self.files)), 4)
            # save the target size of the image
            target_size = img.size
            # open the images and resize them to the target size and convert them to RGB
            img1 = Image.open(os.path.join(self.root, self.files[indices[0]])).convert("RGB").resize(target_size)
            img2 = Image.open(os.path.join(self.root, self.files[indices[1]])).convert("RGB").resize(target_size)
            img3 = Image.open(os.path.join(self.root, self.files[indices[2]])).convert("RGB").resize(target_size)
            img4 = Image.open(os.path.join(self.root, self.files[indices[3]])).convert("RGB").resize(target_size)
            # Call the column_mix function to create a new image by mixing the four images
            return column_mix(img1, img2, img3, img4)
        # If mix_prob is not met, return the original image
        return img

# Adding Gaussian noise to the image
class AddGaussianNoise:
    """
    This class adds Gaussian noise to a tensor image. We can control the mean and standard deviation of the noise.
    The standard deviation is the amount of noise to be added and given by the user. Probability is the chance of adding noise to the image 
    and decide how percentage of images to be augmented.

    """

    # initializing the class with mean, std and p
    def __init__(self, mean=0.0, std=1.0, p=0.3):
        self.mean, self.std, self.p = mean, std, p
     
    # this function it takes a tensor as input and adds Gaussian noise to it
    # if the random number is less than p, it adds noise to the image
    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("AddGaussianNoise expects a tensor input")
        if torch.rand(1).item() < self.p:
            return tensor + torch.randn_like(tensor) * self.std + self.mean
        return tensor
    # this function is used to print the class name and its parameters
    def __repr__(self):
        return (f"{self.__class__.__name__}(mean={self.mean}, "
                f"std={self.std}, p={self.p})")

# Custom dataset class for the melanoma dataset
class MelanomaDataset(Dataset):
    def __init__(self, opt, mode, root, files, classes, transforms_tuple=None):
        """
        This class is the main goal of the module. It is used to load the dataset and apply transformations to the images.
        It is also used to create the dataloaders for training and validation. All modification to the data like Upsampling,
        Downsampling and K-fold cross-validation are done here.
        """
        # read parameters from the config file(.YML)
        self.opt = opt
        # read the mode of the dataset (train or val) 
        self.mode = mode
        # read the root path of the dataset
        self.root = root
        # decide to use metadata or not
        self.use_metadata    = bool(opt['model'].get('use_metadata', False))
        
        # Selecting a subset of data. For quick debugging purposes.
        if self.opt['dataset']['subset'] < 1.0:
            num_samples = int(len(files) * self.opt['dataset']['subset'])
            self.files = files[:num_samples]
            self.classes = classes[:num_samples]
        else:
            self.files = files
            self.classes = classes
        # identify contrastive mode
        self.use_contrastive = (opt['model'].get('loss_function', '') == 'contrastive')

        
        # Build transforms for base and additional augmentations for class 1.
        if transforms_tuple is None:
            self.base_transforms, self.class1_transforms = self.build_transforms()
        else:
            self.base_transforms, self.class1_transforms = transforms_tuple

    def __getitem__(self, item):
        # Load image and label
        image = Image.open(os.path.join(self.root, self.files[item])).convert("RGB")
        label = self.classes[item]
        # if using contrastive mode, always generate 2 views are generated
        if self.use_contrastive:
            # Always generate 2 views if contrastive mode
            image1 = self.apply_transforms(image, label)
            image2 = self.apply_transforms(image, label)
            return (image1, image2), label
        else:
            # passing the image and label to the apply_transforms function for deciding the augmentation approach
            image = self.apply_transforms(image, label)
            if self.use_metadata:
            # metadata was built & attached in the dataloader
                meta = self.metadata[item]
                return (image, meta), label
            else:
                return image, label
            


    def apply_transforms(self, image, label):
        """
        Apply transformations to the image based on its label. If the label is 1, apply class 1 transformations.

        """
        aug = self.opt['dataset'].get('augmentations', {})
        apply_aug_to_all = aug.get('apply_augmentation_to_all', False)

        if (label == 1 or apply_aug_to_all) and self.class1_transforms is not None:
            image = self.class1_transforms(image)
        if self.base_transforms:
            image = self.base_transforms(image)
        return image


    def __len__(self):
        return len(self.files)

    def build_transforms(self):
        """
        Build the transformations for the dataset. This includes base transformations and class-specific augmentations.
        The transformations are applied to the images based on their labels.

        """
        # Get augmentation options
        aug = self.opt['dataset'].get('augmentations', {})

        # Check if augmentations should apply to all classes or only class 1
        apply_aug_to_all = aug.get('apply_augmentation_to_all', False)

        # Base transforms applied to ALL samples (after any class-specific augmentations)
        base_transforms = [
            transforms.Resize(self.opt['dataset']['image_size']),
            transforms.ToTensor()
        ]

        # build noise and normalization transforms
        if aug.get('gaussian_noise', 0) > 0:
            base_transforms.append(
                AddGaussianNoise(
                    mean=0.0,
                    std=aug['gaussian_noise'],
                    p=aug.get('gaussian_noise_prob', 0.3)
                )
            )
        # Add random erasing if specified
        if aug.get('random_erasing', 0) > 0:
            base_transforms.append(transforms.RandomErasing(p=aug['random_erasing'], value='random'))
        # Add normalization
        # Note: The normalization values are based on ImageNet statistics
        base_transforms.append(transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225]))
        base_transforms = transforms.Compose(base_transforms)

        # --- Class 1 augmentations (before ToTensor) or ALL classes depending on setting ---
        augmentation_transforms = []
        if self.mode == "train":
            if aug.get('horizontal_flip', 0) > 0:
                # Randomly flip the image horizontally with a probability that is set in the config file
                augmentation_transforms.append(transforms.RandomHorizontalFlip(p=aug['horizontal_flip']))
            if aug.get('vertical_flip', 0) > 0:
                # Randomly flip the image vertically with a probability that is set in the config file
                augmentation_transforms.append(transforms.RandomVerticalFlip(p=aug['vertical_flip']))
            if aug.get('random_rotation', 0) > 0:
                # Randomly rotate the image with a degree that is set in the config file
                augmentation_transforms.append(transforms.RandomRotation(
                    degrees=aug['random_rotation'],
                    interpolation=InterpolationMode.NEAREST,
                    fill=(255, 255, 255)
                ))
            # Randomly crop the image with a size that is set in the config file    
            if aug.get('random_shear', 0) > 0:
                augmentation_transforms.append(transforms.RandomAffine(degrees=0, shear=aug['random_shear'], fill=(255, 255, 255)))
            if aug.get('shift_vertical', None) is not None:
                # Randomly shift the image vertically with a value that is set in the config file
                augmentation_transforms.append(transforms.RandomAffine(degrees=0, translate=(0, aug['shift_vertical']), fill=(255, 255, 255)))
            if aug.get('shift_horizontal', None) is not None:
                # Randomly shift the image horizontally with a value that is set in the config file
                augmentation_transforms.append(transforms.RandomAffine(degrees=0, translate=(aug['shift_horizontal'], 0), fill=(255, 255, 255)))
            if aug.get('random_zoom', None):
                # Randomly zoom the image with a value that is set in the config file
                # The zoom value is a tuple (zmin, zmax) where zmin and zmax are the minimum and maximum zoom values
                zmin, zmax = aug['random_zoom']
                #make background white
                augmentation_transforms.append(transforms.RandomAffine(degrees=0, scale=(zmin, zmax), fill=(255, 255, 255)))
            if aug.get('color_jitter', 0) > 0:
                # Randomly change the brightness, contrast, saturation and hue of the image with a value that is set in the config file
                cj_value = aug['color_jitter']
                augmentation_transforms.append(transforms.ColorJitter(brightness=cj_value, contrast=cj_value, saturation=cj_value))
            if aug.get('elastic_transform', False) and ElasticTransform is not None:
                # Apply elastic transform with a value that is set in the config file
                # alpha is the intensity of the elastic deformation
                augmentation_transforms.append(ElasticTransform(alpha=50.0))
            if aug.get('image_mix_enabled', False):
                # Apply quadrant mix with a probability that is set in the config file
                mix_prob = aug.get('image_mix_prob', 1.0)
                augmentation_transforms.append(QuadrantMixTransform(mix_prob, self.root, self.files))
        
        augmentation_transforms = transforms.Compose(augmentation_transforms) if augmentation_transforms else None
        # If augmentation_transforms is None, it means no class-specific augmentations are needed
        # If apply_aug_to_all is True, we don't need to return class1_transforms
        if apply_aug_to_all:
            #  if apply to all classes, use in base transform
            if augmentation_transforms:
                full_transform = transforms.Compose([
                    augmentation_transforms,
                    base_transforms
                ])
                return full_transform, None  # No special class1 transforms
            else:
                return base_transforms, None
        else:
            # if only for class 1
            return base_transforms, augmentation_transforms



def stratified_sampler(labels):
    """
    labels: 1-D array or list of 0/1 (or multi‐class) ground‐truths
    returns: a WeightedRandomSampler that draws with replacement,
             using 1/count(class) as the per‐sample weight.
    """
    labels = np.asarray(labels, dtype=int)
    # Count how many of each class
    class_counts = np.bincount(labels)
    # Weight for class i = 1 / count_i
    class_weights = 1.0 / class_counts
    # Weight for each sample = weight of its class
    sample_weights = class_weights[labels]
    sample_weights = torch.from_numpy(sample_weights).float()
    # Draw len(labels) samples each epoch, with replacement
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )


def balanced_val(dataset):
    """
    Create a balanced validation set by randomly sampling equal numbers of class 0 and class 1 samples.

    """
    files = dataset.files
    classes = dataset.classes

    # Separate class 0 and class 1
    class_0 = [(f, c) for f, c in zip(files, classes) if c == 0]
    class_1 = [(f, c) for f, c in zip(files, classes) if c == 1]

    min_class_size = min(len(class_0), len(class_1))

    # Randomly select min_class_size samples from each
    rng = random.Random(42)  # Create independent random generator
    class_0_balanced = rng.sample(class_0, min_class_size)
    class_1_balanced = rng.sample(class_1, min_class_size)

    # Combine and shuffle
    balanced_samples = class_0_balanced + class_1_balanced
    random.shuffle(balanced_samples)

    balanced_files, balanced_classes = zip(*balanced_samples)

    # Create a new dataset with the balanced samples
    new_ds = MelanomaDataset(
        dataset.opt, dataset.mode, dataset.root,
        list(balanced_files), list(balanced_classes),
        transforms_tuple=(dataset.base_transforms, dataset.class1_transforms)
    )

    # ── Propagate metadata from the original ──
    # For each balanced file, find its index in the original dataset.files
    new_ds.metadata = [
        dataset.metadata[ dataset.files.index(fname) ]
        for fname in balanced_files
    ]

    return new_ds

  

def up_sampling(files, classes, oversampling_rate=2):
    """
    By this function we can duplicate the samples of class 1 to balance the dataset
    by the ratio that is set in the config file.

    """

    # Naive random over-sampling: duplicate class 1 samples
    class_0_files = [f for f, label in zip(files, classes) if label == 0]
    class_1_files = [f for f, label in zip(files, classes) if label == 1]
    # Oversample class 1
    # Note: oversampling_rate should be > 1.0
    oversampled_class1 = class_1_files * oversampling_rate
    # Combine class 0 and oversampled class 1
    new_files = class_0_files + oversampled_class1
    # Create new classes list
    new_classes = [0]*len(class_0_files) + [1]*len(oversampled_class1)
    return new_files, new_classes

def down_sampling(files, classes, downsampling_rate=1.0):
    """
    By this function we can downsample the samples of class 0 to balance the dataset.
    The ratio is set in the config file and should be between 0 and 1.

    """
    # Randomly downsample majority class (class 0)
    class_0 = [(f, c) for f, c in zip(files, classes) if c == 0]
    class_1 = [(f, c) for f, c in zip(files, classes) if c == 1]
    # Downsample class 0
    # Note: downsampling_rate should be < 1.0
    num_to_keep = int(len(class_0) * downsampling_rate)
    # Randomly select num_to_keep samples from class 0
    class_0_downsampled = random.sample(class_0, num_to_keep)
    # Combine downsampled class 0 and class 1
    combined = class_0_downsampled + class_1
    # Shuffle the combined list
    random.shuffle(combined)
    new_files, new_classes = zip(*combined)
    return list(new_files), list(new_classes)

def melanoma_train_dataloaders(opt):
    """
    This function creates the dataloaders for the training and validation sets.
    It uses the MelanomaDataset class to load the data and apply transformations.
    It also handles the K-fold cross-validation and stratified sampling if specified in the config file.
    The function returns the dataloaders for training, validation and balanced validation sets.
    
    """
    #TODO : Need more comments but I am not familar with this part

    dataset = pd.read_csv(opt['dataset']['dataset_train_csv'])
    # 1) Drop rows with missing metadata
    if opt['model']['use_metadata']:
        dataset = dataset.dropna(subset=['sex','age_approx','anatom_site_general_challenge'])
        dataset = dataset[dataset['sex'].str.strip() != '']
        dataset = dataset[dataset['anatom_site_general_challenge'].str.strip() != '']
    # dataset = dataset[dataset['age_approx'].str.strip()!='']
    valid_groups = [0,1,2,3,4,5,6,7,8,9,10,11]
    # If use_groupkfold is set to True, we will use GroupKFold for cross-validation
    dataset= dataset[dataset['tfrecord'].isin(valid_groups)].reset_index(drop=True)
    files = dataset['image_name'] + '.jpg'
    classes = dataset['target'].values

    # Note this is a 4-fold config to maximise the use of available data in a leak free fashion.
    # We use soft voting which means the number of folds does not have to be an odd number.
    CUSTOM_FOLDS = [
        {'train': [0, 1, 2, 3, 4, 5, 6, 7, 8],  'val': [9, 10, 11]},
        {'train': [0, 1, 2, 3, 4, 5, 9, 10, 11],  'val': [6, 7, 8]},
        {'train': [0, 1, 2, 6, 7, 8, 9, 10, 11],  'val': [3, 4, 5]},
        {'train': [3, 4, 5, 6, 7, 8, 9, 10, 11],  'val': [0, 1, 2]},
    ]
    
    # ── K-Fold Cross Validation ──
    # If use_groupkfold is set to True, we will use GroupKFold for cross-validation
    if opt['dataset'].get('use_groupkfold', False):
        if opt['model']['use_metadata']:
            sex_map  = {'male': 0, 'female': 1}
            site_map = {
                s: i for i, s in
                enumerate(sorted(dataset['anatom_site_general_challenge'].unique()))
            }
        fold_loaders = []

        # Create a GroupKFold object with the number of splits
        for count, fold_cfg in enumerate(CUSTOM_FOLDS):
            tr_groups, val_groups = fold_cfg['train'], fold_cfg['val']
        
            # Indices for train and validation groups
            # Note: This is a custom split based on the tfrecord groups    
            train_mask = dataset['tfrecord'].isin(tr_groups)
            val_mask   = dataset['tfrecord'].isin(val_groups)
            # specify the train files and classes from the dataset    
            train_files   = (dataset.loc[train_mask, 'image_name'] + '.jpg').tolist()
            train_classes =  dataset.loc[train_mask, 'target'].values
    
            # specify the val files and classes from the dataset
            # Note: This is a custom split based on the tfrecord groups
            val_files   = (dataset.loc[val_mask, 'image_name'] + '.jpg').tolist()
            val_classes =  dataset.loc[val_mask, 'target'].values

            # ── Build metadata for this fold ──
            if opt['model']['use_metadata']:
                train_df = dataset.loc[train_mask].reset_index(drop=True)
                val_df   = dataset.loc[val_mask].reset_index(drop=True)

                train_meta = utils.build_metadata(train_df, sex_map, site_map)
                val_meta   = utils.build_metadata(val_df,   sex_map, site_map)
            else:
                train_meta = [None] * len(train_files)
                val_meta   = [None] * len(val_files)

            # ─────── oversampling / down‑sampling exactly as before ───────
            # Note: downsampling_rate should be < 1.0
            if opt['dataset'].get('downsampling_rate', 1.0) < 1.0:
                train_files, train_classes = down_sampling(
                    train_files, train_classes,
                    opt['dataset']['downsampling_rate']
                )
              
            # Note: oversampling_rate should be > 1.0
            if opt['dataset'].get('oversampling_rate', 1.0) > 1.0:
                train_files, train_classes = up_sampling(
                    train_files, train_classes,
                    opt['dataset']['oversampling_rate']
                )
    
            # ─────── build MelanomaDataset objects ───────
            train_ds = MelanomaDataset(opt, 'train',
                                       opt['dataset']['dataset_train_path'],
                                       train_files, train_classes)
            train_ds.metadata = train_meta
            val_ds   = MelanomaDataset(opt, 'val',
                                       opt['dataset']['dataset_val_path'],
                                       val_files, val_classes)
            val_ds.metadata = val_meta
    
            # ─────── DataLoaders (with optional stratified sampler) ───────
            if opt['dataset'].get('use_stratified_sampler', False):
                sampler = stratified_sampler(train_ds.classes)
                train_loader = DataLoader(train_ds,
                                          batch_size=opt['dataset']['batch_size'],
                                          sampler=sampler, num_workers=2)
            else:
                train_loader = DataLoader(train_ds,
                                          batch_size=opt['dataset']['batch_size'],
                                          shuffle=True, num_workers=2)
    
            val_loader = DataLoader(val_ds,
                                    batch_size=opt['dataset']['batch_size'],
                                    shuffle=False, num_workers=2)
    
            # balanced validation set (same helper you already had)
            val_bal_ds = balanced_val(val_ds)
            val_loader_bal = DataLoader(val_bal_ds,
                                        batch_size=opt['dataset']['batch_size'],
                                        shuffle=False, num_workers=2)
    
            fold_loaders.append({
                'fold': count+1,                 # (optional) keeps track
                'train_loader': train_loader,
                'val_loader':   val_loader,
                'val_loader_balanced': val_loader_bal,
            })

            print(f"Fold #{count+1} | Train-groups: {tr_groups} | Val-groups: {val_groups}")
            print(f"Fold #{count+1} | Train balance:");  utils.check_dataset_balance(train_ds)
            print(f"Fold #{count+1} | Natural-Val balance:");    utils.check_dataset_balance(val_ds)
            print(f"Fold #{count+1} | Balanced‑Val:");utils.check_dataset_balance(val_bal_ds)
    
        return fold_loaders
# ------------------------------------ Simple split by tfrecord ------------------------------------
    else:
        # Simple split by tfrecord: groups 0-9 for training, 10-11 for validation
        train_df = dataset[dataset['tfrecord'].isin(range(0, 10))]
        val_df = dataset[dataset['tfrecord'].isin([10, 11])]

        # ── Drop rows with missing metadata ──
        if opt['model']['use_metadata']:
            # dropna as before…
            sex_map  = {'male':0,'female':1}
            site_map = {s:i for i,s in 
                        enumerate(sorted(dataset['anatom_site_general_challenge'].unique()))}

            train_meta = utils.build_metadata(train_df, sex_map, site_map)
            val_meta   = utils.build_metadata(val_df,   sex_map, site_map)
        else:
            train_meta = [None] * len(train_df)
            val_meta   = [None] * len(val_df)


        # ── Specify train/val files and classes ──
        train_files = (train_df['image_name'] + '.jpg').tolist()
        # Note: This is a custom split based on the tfrecord groups
        train_classes = train_df['target'].tolist()

        # ── Specify val files and classes ──
        val_files = (val_df['image_name'] + '.jpg').tolist()
        val_classes = val_df['target'].tolist()
        
        # ── Oversampling / downsampling ──
        if opt['dataset'].get('oversampling_rate', 1.0) > 1.0:
            oversampling_rate = opt['dataset'].get('oversampling_rate', 1.0)
            print("Applying upsampling to the training set with rate", oversampling_rate)
            train_files, train_classes = up_sampling(train_files, train_classes, oversampling_rate)
        
        # ── Downsampling ──
        if opt['dataset'].get('downsampling_rate', 1.0) < 1.0:
            downsampling_rate = opt['dataset'].get('downsampling_rate', 1.0)
            print("Applying downsampling to the training set with rate", downsampling_rate)
            train_files, train_classes = down_sampling(train_files, train_classes, downsampling_rate)
        print(f"After downsampling: {len(train_files)}")

        # ── Build MelanomaDataset objects ──
        train_dataset = MelanomaDataset(opt, 'train', opt['dataset']['dataset_train_path'], train_files, train_classes)
        train_dataset.metadata = train_meta        
        val_dataset = MelanomaDataset(opt, 'val', opt['dataset']['dataset_val_path'], val_files, val_classes)
        val_dataset.metadata   = val_meta

        # ── DataLoaders (with optional stratified sampler) ──
        # If use_stratified_sampler is set to True, we will use stratified sampling
        if opt['dataset'].get('use_stratified_sampler', False):
            sampler = stratified_sampler(train_dataset.classes)
            train_loader = DataLoader(train_dataset, batch_size=opt['dataset']['batch_size'], sampler=sampler, num_workers=2)

        else:
            # If not using stratified sampling, just shuffle the dataset
            train_loader = DataLoader(train_dataset, batch_size=opt['dataset']['batch_size'], shuffle=True, num_workers=2)

        # Create DataLoader for validation set
        val_loader = DataLoader(val_dataset, batch_size=opt['dataset']['batch_size'],
                                shuffle=False, num_workers=2)
        # Create DataLoader for balanced validation set
        val_dataset_balanced = balanced_val(val_dataset)
        # Create DataLoader for balanced validation set
        val_loader_balanced = DataLoader(
            val_dataset_balanced, batch_size=opt['dataset']['batch_size'],
            shuffle=False, num_workers=2)

        print("Train Balance:")
        utils.check_dataset_balance(train_dataset)
        print("Val Balance:")
        utils.check_dataset_balance(val_dataset)
        print("Balanced Val Balance:")
        utils.check_dataset_balance(val_dataset_balanced)

        return train_loader, val_loader, val_loader_balanced


def melanoma_test_dataloaders(opt):
    """
    This function creates the dataloader for the test set.
    It uses the MelanomaDataset class to load the data and apply transformations.

    """
    # 1) Load Dataset:
    dataset = pd.read_csv(opt['dataset']['dataset_test_csv'])

    # 2) Drop rows with missing metadata
    if opt['model']['use_metadata']:
        dataset = dataset.dropna(subset=['sex','age_approx','anatom_site_general_challenge'])
        dataset = dataset[dataset['sex'].str.strip()!='']
        dataset = dataset[dataset['anatom_site_general_challenge'].str.strip()!='']
    if opt['model']['use_metadata']:
        sex_map  = {'male':0,'female':1}
        site_map = {s:i for i,s in 
                    enumerate(sorted(dataset['anatom_site_general_challenge'].unique()))}

        meta_test = utils.build_metadata(dataset, sex_map, site_map)
    else:
        meta_test = [None] * len(dataset)

    files = dataset['image_name'].values + '.jpg'       # Images need .jpg to be found

    # Check the dataset for missing labels
    if 'target' not in dataset:
        classes = np.full((len(dataset),), -1)
    else:
        # If the target column is present, use it
        classes = dataset['target'].values

    # if we are missing labels we need to tell the main loop that we can't evaluate our own predictions
    if np.min(classes) < 0:
        # This is a test set with no labels
        predictmode = True
    else:
        # This is a test set with labels
        predictmode = False

    # ── Build MelanomaDataset object ──
    test_dataset = MelanomaDataset(
        opt,
        'val',
        opt['dataset']['dataset_test_path'], files, classes
    )
    test_dataset.metadata = meta_test

    # ── DataLoader ──
    test_loader = DataLoader(
        test_dataset,
        batch_size=opt['dataset']['batch_size'],
        shuffle=False,
        num_workers=2
    )

    start_time = time.time()
    total_images = 0

    # ── Count the number of images in the test set ──
    for images, _ in test_loader:
        # unpack metadata-augmented batches just like in train/val
        if isinstance(images, (list, tuple)):
            images = images[0]
        total_images += images.size(0)

    duration = time.time() - start_time
    fps = total_images / duration if duration > 0 else 0
    print(f"[Test Loader] Processed {total_images} images in {duration:.2f} seconds -> {fps:.2f} FPS")


    return predictmode, test_loader
