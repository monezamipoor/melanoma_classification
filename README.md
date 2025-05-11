# EEEM068 - Group 13 - Melanona Classification
This git repository hold the codebase and example notebooks designed to be run in Google Colab for a deep learning melanoma classifcation system.

# Quick Start
Two notebooks are provided:

- <NOTEBOOK 1 LINK> - A self-contained set of examples for running training and test.
- <NOTEBOOK 2 LINK> - Covers explainability tests 

Both notebooks can be executed on free-tier [Google Colab](https://colab.research.google.com/)

# Python Files and Architecture
![](https://github.com/monezamipoor/melanoma_classification/blob/main/ml-coursework.png?raw=true)

## Key Files
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
