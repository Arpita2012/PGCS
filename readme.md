# Code for Perceptually Guided Coreset Selection(PGCS)
Work accepted in ICIP 2025

Codebase to be released soon!


### Edit dataset-meta.json to provide path to dataset root folder 
IQA Dataset publicly available on https://huggingface.co/datasets/chaofengc/IQA-PyTorch-Datasets 

### Create the environment from the environment.yml file:
conda env create -f environment.yml

### Edit iqa_config.py
Provide dataset name and dataset fraction. All hyperparameters can be configured from iqa_config.py

### LIQE Checkpoint Download 
Link: https://drive.google.com/file/d/1GoKwUKNR-rvX11QbKRN8MuBZw2hXKHGh/view?usp=sharing 
Move LIQE.pt inside iqa_architecture/chekpoint/

### Select coreset
python iqa_select_idx.py

This will save ids of images selected as coreset in 'indices' folder

### Train on IQA Architecture
python iqa_train.py

### Check training logs
output/log/<dataset_name>/<IQA_architecture/Coreset_Selection_Method/Dataset_Fraction>



