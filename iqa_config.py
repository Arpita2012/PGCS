import torch
import os
import numpy as np
import random
import json
import argparse

class Config(dict): 
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

    @classmethod
    def load(cls, file):
        with open(file, 'r') as f:
            config = json.loads(f.read())
            return Config(config)
    
def set_config():
# config file
    config = Config({
        # dataset path
        "dataset_name": "AGIQA_3K", 

        "device": "cuda:0",
        
        # optimization
        "batch_size": 2,
        "learning_rate": 1e-5,
        "weight_decay": 1e-5,
        "n_epoch": 25,
        "val_freq": 1,
        "T_max": 50,
        "eta_min": 0,
        "num_avg_val": 1, 
        "num_workers": 8,
        
        # data
        "dataset_meta_file":'dataset_info/dataset-meta.json',
        "split_seed": 20,
     
        "crop_size": 224,
        "prob_aug": 0.7,

        # model
        "patch_size": 8,
        "img_size": 224,
        "embed_dim": 768,
        "dim_mlp": 768,
        "num_heads": [4, 4],
        "window_size": 4,
        "depths": [2, 2],
        "num_outputs": 1,
        "num_tab": 2,
        "scale": 0.8,
        "reg_penalty": 1e-4, 
        
        # load & save checkpoint
        "model_name": "maniqa",
        
        "ckpt_path": "./output/models/",  # directory for saving checkpoint
        "log_path": "./output/log/",
        "log_file": ".log",
        "tensorboard_path": "./output/tensorboard/", 
        "early_stop": 8, # early stopping if the val main_score does not increase for 8 epochs continuously

        #Coreset settings
        "coreset_as_dataset_fraction":0.15, #0.01 0.05 0.1 0.3 0.5 0.7 0.9 0.95
        "coreset_size": None, 
       
        "coreset_method":    'PGCS' ,
        "index_folder" : "./indices",
        "seed": 20,


        #image_embeddings save
       
        "feature_save_path": "./features",
        "feature_save_file": "image_features.pkl", #specify name of which layers features were extracted
        "distance_file_name": "distance.bin", 
        #DCQ_CS Settings
        "cluster_save_path": "./clusters",
        "cluster_save_file": "image_clusters.pkl", #specify name of NAME OF FILE DEPENDING UPON PARAMETER OF CLUSTERING METHOD
        
        "partition_save_path": "./partition",
        "partition_save_file": "image_partition.pkl", #specify name of NAME OF FILE DEPENDING UPON PARAMETER OF CLUSTERING METHOD
        "partioningAlgo":  "CoveragePrecedencePartitioning", #  "RandomPartitioning" , # "Kmeans",
        "projectionOperator":"TSNE",
        "prjection_hyper":3,
        "partion_count":5,
        "imageEncoder":"liqe"



    })
    

    return config



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coreset_as_dataset_fraction", type=float, default=None, help="coreset size")  
    parser.add_argument("--coreset_size", type=int, default=None, help="coreset size")    
    parser.add_argument("--coreset_method", type=str, default=None, help="coreset construction method")
    parser.add_argument("--dataset_name", type=str, default=None, help="dataset to train on")
    parser.add_argument("--cross_dataset", type=str, default=None, help="dataset test set to evaluate on")
     
   
    args = parser.parse_args()
    return args


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
