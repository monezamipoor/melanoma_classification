# EEEM068 - Group 13 - Melanona Classification
This git repository hold the codebase and example notebooks designed to be run in Google Colab for a deep learning melanoma classifcation system.

# Quick Start
Two notebooks are provided:

- <NOTEBOOK 1 LINK> - A self-contained set of examples for running training and test.
- <NOTEBOOK 2 LINK> - Covers explainability tests 

Both notebooks can be executed on free-tier [Google Colab](https://colab.research.google.com/) with a T4 GPU. The notebooks are designed to:
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
| File | Purpose |
| ----------- | ----------- |
| | |

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

