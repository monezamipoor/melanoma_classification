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

# Creating ColumnMix Augmentation which creates an image consisted of four stripes from four images
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

# Creating QuadrantMix Augmentation which creates mosaics of four images in a grid
class QuadrantMixTransform:
    def __init__(self, mix_prob, root, files):
        self.mix_prob = mix_prob
        self.root = root
        self.files = files

    def __call__(self, img):
        # With probability mix_prob, perform quadrant mix.
        if random.random() < self.mix_prob and len(self.files) >= 4:
            indices = random.sample(range(len(self.files)), 4)
            target_size = img.size
            img1 = Image.open(os.path.join(self.root, self.files[indices[0]])).convert("RGB").resize(target_size)
            img2 = Image.open(os.path.join(self.root, self.files[indices[1]])).convert("RGB").resize(target_size)
            img3 = Image.open(os.path.join(self.root, self.files[indices[2]])).convert("RGB").resize(target_size)
            img4 = Image.open(os.path.join(self.root, self.files[indices[3]])).convert("RGB").resize(target_size)
            return column_mix(img1, img2, img3, img4)
        return img

class AddGaussianNoise:
    def __init__(self, mean=0.0, std=1.0, p=0.3):
        self.mean, self.std, self.p = mean, std, p

    def __call__(self, tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("AddGaussianNoise expects a tensor input")
        if torch.rand(1).item() < self.p:
            return tensor + torch.randn_like(tensor) * self.std + self.mean
        return tensor

    def __repr__(self):
        return (f"{self.__class__.__name__}(mean={self.mean}, "
                f"std={self.std}, p={self.p})")

class MelanomaDataset(Dataset):
    def __init__(self, opt, mode, root, files, classes, transforms_tuple=None):
        """
        Args:
            opt (dict): Options dictionary.
            mode (str): "train" or "val".
            root (str): Root directory for images.
            files (list): List of image file names.
            classes (list): List of class labels.
            transforms_tuple (tuple, optional): A tuple (base_transforms, class1_transforms). If None, they are built from opt.
            subset (float): Fraction of data to use.
        """
        self.opt = opt
        self.mode = mode
        self.root = root
        
        # Selecting a subset of data. For quick debugging purposes.
        if self.opt['dataset']['subset'] < 1.0:
            num_samples = int(len(files) * self.opt['dataset']['subset'])
            self.files = files[:num_samples]
            self.classes = classes[:num_samples]
        else:
            self.files = files
            self.classes = classes

        # Build transforms for base and additional augmentations for class 1.
        if transforms_tuple is None:
            self.base_transforms, self.class1_transforms = self.build_transforms()
        else:
            self.base_transforms, self.class1_transforms = transforms_tuple

    def __getitem__(self, item):
        # Load the image as a PIL image.
        image = Image.open(os.path.join(self.root, self.files[item])).convert("RGB")
        label = self.classes[item]

        # For class 1 samples, apply the extra augmentation before the base transforms.
        if label == 1 and self.class1_transforms is not None:
            image = self.class1_transforms(image)

        # Now apply the base transforms (e.g., resize, ToTensor, Normalize) to every sample.
        if self.base_transforms:
            image = self.base_transforms(image)

        return image, label

    def __len__(self):
        return len(self.files)

    def build_transforms(self):
        # Get augmentation options
        aug = self.opt['dataset'].get('augmentations', {})

        # --- Base transforms applied to ALL samples (after any class-specific augmentations) ---
        base_transforms = [
            transforms.Resize(self.opt['dataset']['image_size']),
            transforms.ToTensor()
        ]

        # Tensor-level augmentations (must happen AFTER ToTensor)
        if aug.get('gaussian_noise', 0) > 0:
            base_transforms.append(
                AddGaussianNoise(
                    mean=0.0,
                    std=aug['gaussian_noise'],
                    p=aug.get('gaussian_noise_prob', 0.3)
                )
            )
        if aug.get('random_erasing', 0) > 0:
            base_transforms.append(transforms.RandomErasing(p=aug['random_erasing'], value='random'))

        base_transforms.append(transforms.Normalize([0.485, 0.456, 0.406],
                                                    [0.229, 0.224, 0.225]))
        base_transforms = transforms.Compose(base_transforms)

        # --- Class 1 augmentations (before ToTensor, only for class 1 in training) ---
        class1_transforms_list = []
        if self.mode == "train":

            if aug.get('horizontal_flip', 0) > 0:
                class1_transforms_list.append(transforms.RandomHorizontalFlip(p=aug['horizontal_flip']))
            if aug.get('vertical_flip', 0) > 0:
                class1_transforms_list.append(transforms.RandomVerticalFlip(p=aug['vertical_flip']))
            if aug.get('random_rotation', 0) > 0:
                class1_transforms_list.append(transforms.RandomRotation(
                    degrees=aug['random_rotation'],
                    interpolation=InterpolationMode.NEAREST,
                    fill=(255, 255, 255)
                ))
            if aug.get('random_shear', 0) > 0:
                class1_transforms_list.append(transforms.RandomAffine(degrees=0, shear=aug['random_shear'], fill=(255,255,255)))
            if aug.get('shift_vertical', None) is not None:
                class1_transforms_list.append(transforms.RandomAffine(degrees=0, translate=(0, aug['shift_vertical']), fill=(255,255,255)))
            if aug.get('shift_horizontal', None) is not None:
                class1_transforms_list.append(transforms.RandomAffine(degrees=0, translate=(aug['shift_horizontal'], 0), fill=(255,255,255)))
            if aug.get('random_zoom', None):
                zmin, zmax = aug['random_zoom']
                class1_transforms_list.append(transforms.RandomAffine(degrees=0, scale=(zmin, zmax), fill=(255,255,255)))
            if aug.get('color_jitter', 0) > 0:
                cj_value = aug['color_jitter']
                class1_transforms_list.append(transforms.ColorJitter(brightness=cj_value, contrast=cj_value, saturation=cj_value))
            if aug.get('elastic_transform', False) and ElasticTransform is not None:
                class1_transforms_list.append(ElasticTransform(alpha=50.0))
            if aug.get('image_mix_enabled', False):
                mix_prob = aug.get('image_mix_prob', 1.0)
                class1_transforms_list.append(QuadrantMixTransform(mix_prob, self.root, self.files))

        class1_transforms = transforms.Compose(class1_transforms_list) if class1_transforms_list else None

        return base_transforms, class1_transforms



# TODO this needs implementing properly and testing?
def stratified_sampler(classes):

    # dataset = pd.read_csv(opt['dataset']['dataset_train_csv'])
    # classes = list(dataset['target'].values)

    # If use_stratified, use some positive samples in each batch
    # If oversampling, increase weights for minority classes
    # Create and return sampler uisng params
    
    classes_arr = np.array(classes)
    
    # Get unique classes and their respective counts
    class_counts = np.bincount(classes)

    # This will compute the weight for each class by counting the number of samples in each class
    class_weights = 1. / class_counts

    # it will assign a weight to each sample based on its class
    sample_weights = [class_weights[label] for label in classes]

    
    # Create the WeightedRandomSampler. Replacement=True allows oversampling.
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

def balanced_val(dataset):
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

    return MelanomaDataset(
        dataset.opt, dataset.mode, dataset.root,
        list(balanced_files), list(balanced_classes),
        transforms_tuple=(dataset.base_transforms, dataset.class1_transforms)
    )

  

def up_sampling(files, classes, oversampling_rate=2):

    # Naive random over-sampling: duplicate class 1 samples
    class_0_files = [f for f, label in zip(files, classes) if label == 0]
    class_1_files = [f for f, label in zip(files, classes) if label == 1]
    oversampled_class1 = class_1_files * oversampling_rate
    new_files = class_0_files + oversampled_class1
    new_classes = [0]*len(class_0_files) + [1]*len(oversampled_class1)
    return new_files, new_classes

def down_sampling(files, classes, downsampling_rate=1.0):
    # Randomly downsample majority class (class 0)
    class_0 = [(f, c) for f, c in zip(files, classes) if c == 0]
    class_1 = [(f, c) for f, c in zip(files, classes) if c == 1]
    num_to_keep = int(len(class_0) * downsampling_rate)
    class_0_downsampled = random.sample(class_0, num_to_keep)
    combined = class_0_downsampled + class_1
    random.shuffle(combined)
    new_files, new_classes = zip(*combined)
    return list(new_files), list(new_classes)

def melanoma_train_dataloaders(opt):
    dataset = pd.read_csv(opt['dataset']['dataset_train_csv'])
    files = dataset['image_name'].values + '.jpg'
    classes = dataset['target'].values

    if opt['dataset'].get('use_groupkfold', False):
        test_dataset = dataset[dataset['tfrecord'].isin([12, 13, 14])]
        train_dataset = dataset[dataset['tfrecord'].isin([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])]
        oversampling_rate = opt['dataset'].get('oversampling_rate', 1.0)
        all_train_files = train_dataset['image_name'].values + '.jpg'
        all_train_classes = train_dataset['target'].values
        all_groups = train_dataset['tfrecord'].values
        n_splits = opt['dataset'].get('n_splits', 3)
        group_kfold = GroupKFold(n_splits=n_splits)
        fold_loaders = []
        
        for fold, (train_idx, val_idx) in enumerate(group_kfold.split(all_train_files, all_train_classes, groups=all_groups)):
            train_files_fold = all_train_files[train_idx]
            val_files_fold = all_train_files[val_idx]
            train_classes_fold = all_train_classes[train_idx]
            val_classes_fold = all_train_classes[val_idx]
            downsampling_rate = opt['dataset'].get('downsampling_rate', 1.0)
            if downsampling_rate < 1.0:
                print(f"Applying downsampling to the training set for fold {fold} with rate {downsampling_rate}")
                train_files_fold, train_classes_fold = down_sampling(train_files_fold, train_classes_fold, downsampling_rate)
            if opt['dataset'].get('oversampling_rate', 1.0) > 1.0:
                oversampling_rate = opt['dataset'].get('oversampling_rate', 1.0)
                print("Applying upsampling to the training set for fold", fold, "with rate", oversampling_rate)
                train_files_fold, train_classes_fold = up_sampling(train_files_fold, train_classes_fold, oversampling_rate)

            train_dataset_fold = MelanomaDataset(opt, 'train', opt['dataset']['dataset_train_path'],
                                                  train_files_fold, train_classes_fold)
            val_dataset_fold = MelanomaDataset(opt, 'val', opt['dataset']['dataset_val_path'],
                                                val_files_fold, val_classes_fold)

            if opt['dataset'].get('use_stratified_sampler', False):
                sampler = stratified_sampler(train_dataset_fold.classes)
                train_loader_fold = DataLoader(train_dataset_fold, batch_size=opt['dataset']['batch_size'], sampler=sampler, num_workers=2)
                
            else:
                train_loader_fold = DataLoader(train_dataset_fold, batch_size=opt['dataset']['batch_size'],
                                               shuffle=True, num_workers=2)

            val_loader_fold = DataLoader(val_dataset_fold, batch_size=opt['dataset']['batch_size'],
                                         shuffle=False, num_workers=2)
            val_dataset_fold_balanced = balanced_val(val_dataset_fold)
            val_loader_fold_balanced = DataLoader(
                val_dataset_fold_balanced, batch_size=opt['dataset']['batch_size'],
                shuffle=False, num_workers=2)
            
            fold_loaders.append({
                'fold': fold,
                'train_loader': train_loader_fold,
                'val_loader': val_loader_fold,
                'val_loader_balanced': val_loader_fold_balanced
            })

            print("Fold ", str(fold), " Train Balance:")
            utils.check_dataset_balance(train_dataset_fold)
            print("Fold ", str(fold), " Val Balance:")
            utils.check_dataset_balance(val_dataset_fold)
            print("Fold ", str(fold), " Balanced_Val Balance:")
            utils.check_dataset_balance(val_dataset_fold_balanced)

        return fold_loaders
    else:
        train_files, val_files, train_classes, val_classes = train_test_split(
            files, classes, train_size=0.8, test_size=0.2, stratify=classes, random_state=42
        )
        print(f"Original training set size: {len(train_files)}")
        if opt['dataset'].get('oversampling_rate', 1.0) > 1.0:
            oversampling_rate = opt['dataset'].get('oversampling_rate', 1.0)
            print("Applying upsampling to the training set with rate", oversampling_rate)
            train_files, train_classes = up_sampling(train_files, train_classes, oversampling_rate)
        if opt['dataset'].get('downsampling_rate', 1.0) < 1.0:
            downsampling_rate = opt['dataset'].get('downsampling_rate', 1.0)
            print("Applying downsampling to the training set with rate", downsampling_rate)
            train_files, train_classes = down_sampling(train_files, train_classes, downsampling_rate)
        print(f"After downsampling: {len(train_files)}")
        train_dataset = MelanomaDataset(opt, 'train', opt['dataset']['dataset_train_path'], train_files, train_classes)
        val_dataset = MelanomaDataset(opt, 'val', opt['dataset']['dataset_val_path'], val_files, val_classes)
        
        if opt['dataset'].get('use_stratified_sampler', False):
            sampler = stratified_sampler(train_dataset.classes)
            train_loader = DataLoader(train_dataset, batch_size=opt['dataset']['batch_size'], sampler=sampler, num_workers=2)

        else:
            train_loader = DataLoader(train_dataset, batch_size=opt['dataset']['batch_size'], shuffle=True, num_workers=2)

        val_loader = DataLoader(val_dataset, batch_size=opt['dataset']['batch_size'],
                                shuffle=False, num_workers=2)
        val_dataset_balanced = balanced_val(val_dataset)
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

    dataset = pd.read_csv(opt['dataset']['dataset_test_csv'])

    files = dataset['image_name'].values + '.jpg'       # Images need .jpg to be found

    if 'target' not in dataset:
        classes = np.full((len(dataset),), -1)
    else:
        classes = dataset['target'].values

    # if we are missing labels we need to tell the main loop that we can't evaluate our own predictions
    if np.min(classes) < 0:
        predictmode = True
    else:
        predictmode = False

    test_dataset = MelanomaDataset(
        opt,
        'val',
        opt['dataset']['dataset_test_path'], files, classes
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=opt['dataset']['batch_size'],
        shuffle=False,
        num_workers=2
    )

    start_time = time.time()
    total_images = 0

    for images, _ in test_loader:
        total_images += images.size(0)

    duration = time.time() - start_time
    fps = total_images / duration if duration > 0 else 0
    print(f"[Test Loader] Processed {total_images} images in {duration:.2f} seconds -> {fps:.2f} FPS")


    return predictmode, test_loader

