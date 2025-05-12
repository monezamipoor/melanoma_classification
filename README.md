# EEEM068 - Group 13 - Melanona Classification
This git repository hold the codebase and example notebooks designed to be run in Google Colab for a deep learning melanoma classifcation system.

# Quick Start
Two notebooks are provided:

- <NOTEBOOK 1 LINK> - A self-contained set of examples for running training and test.
- <NOTEBOOK 2 LINK> - Covers explainability tests 

NOTE - WHEN WE PUT THESE LINKS IN MAKE SURE IT CAN BE ACCESSED AS A DIRECT COLAB LINK TO GITHUB. I.E.
'https://colab.research.google.com/github/monezamipoor/melanoma_classification/blob/main/<NOTEBOOK 1>.ipynb'

Both notebooks can be executed on free-tier [Google Colab](https://colab.research.google.com/) with a T4 GPU. The notebook is designed to:
1. Automatically clone this git repo for execution
2. Install requirements.txt dependancies
3. Download and unzip the melanoma training set from a google drive location (no login required)
4. List a set or pre-configured tests that can be executed independently

# Execution Notes
- All executions of training contain validation and test loops.
- Results are logged to a new directory named `<YML>-<DATE>`. Path to this directory is defined in each YML file.
- Within the log directory:
```
log_test_balanced.csv - Metric outputs for the balanced test data set (1:1 class ratio)
log_test_natural.csv - Metric outputs for the natural test data set (50:1 class ratio)
log_val_balanced.csv - Valdation stats per epoch for balanced val set
log_val_natural.csv - Valdation stats per epoch for natural val set
ROC__xxxx.png - ROC curves for balanced / nautral tests
confusion_matrix_xxxxxxxx.png - Confusion matrices for balanced/natural tests
```

## Command Line args
`python main.py -o <CONFIG>.yml`

All executions require a named YML file, residing in the /options folder. Supplied with -o at the command line.

## YAML Examples

All contained within /options. The notebook contains all these as executable examples.

| YML File | Purpose / Major changes|
| ----------- | ----------- |
| 1-Base | Pretrained Effnetb0, bce no loss weighting, adam + cosine, LR 0.0001, BS 32 |
| 2-Hyperparameters | adamw + LR 0.001 |
| 3-Sampling | downsampling to 10% class 0, oversampling 500% class 1 |
| 4-Augmentations | Augmentations adjustments to vflip, jitter |
| 5-Loss | BCE + Triplet loss example |
| 6-Model | Resnet50 example |
| 7-Kfold | Folding with comfig files 1-5 optimisations |
| 8-a-Hybrid | CNN only test as a baseline for hybrid test 8-b (same HPs, no transformers) |
| 8-b-Hybrid | Hybrid model test. 128 dims, 4 heads, 2 layers |
| 9-Metadata | Test with patient site, age and gender metadata |
| 10-Contrastive | Contrastive loss example |

## YAML Config Options
Most config is balanced and tailored for the test it is designed to execute. It is recommended to use the named YMLs for the types of test they run (e.g. hybrid or focal loss). However there are a number of general parameters that might be of use to change to affect the pace or outcome of all tests in the provided YMLs:

| What? | Where? | Type |
| ----------- | ----------- | ----------- |
| Control # epochs | training: epochs: | int |
| General LR | training: learning_rate: | float |
| Batch Size | dataset: batch_size: | int |
| Augmentations | dataset: augmentations: (various) | N/A |
| Log directory | testing: log_dir: | string |
| Enable folding/CV | dataset: use_groupfold: | bool |

# Python Files and Architecture
![](https://github.com/monezamipoor/melanoma_classification/blob/main/ml-coursework.png?raw=true)

## Key Python Files
| File | Purpose |
| ----------- | ----------- |
| contrastive_svm.py | Contrastive learning and loss routines |
| data.py | Dataloader and augmentation |
| explainability.py | Explainability models for image feature importance |
| loss.py | Loss function implementations |
| main.py | Contains the main() method that executes the train, validate and test loops |
| model.py | CNN model and forward pass |
| model_hybrid.py | Hybrid CNN/Transformer implementation |
| utils.py | Various utilities and file store management routines |

