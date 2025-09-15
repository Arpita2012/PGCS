from iqa_config import set_config, parse_args, setup_seed
import json
import pickle
import os
import numpy as np
import random
import sys


# Add the folder to the Python path
folder_path = "./src" # os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if folder_path not in sys.path:
    sys.path.insert(0, folder_path)
# print(sys.path)

from src.pgcs import *
from iqa_dataset import get_dataset
from torch.utils.data import DataLoader

config=set_config()
setup_seed(config.seed)

##### Useful when running experiments in batch mode
args = parse_args()
if args.coreset_as_dataset_fraction is not None:
    config.coreset_as_dataset_fraction = args.coreset_as_dataset_fraction

if args.coreset_size is not None:
    config.coreset_size = args.coreset_size

if args.coreset_method is not None:
    config.coreset_method = args.coreset_method

if args.dataset_name is not None:
    config.dataset_name = args.dataset_name


with open(config.dataset_meta_file, 'r') as f:
    dataset_meta = json.load(f)


dataset_name = config.dataset_name
dataset_info = dataset_meta[dataset_name]


train_dataset = get_dataset (dataset_name, dataset_info, type = 'train',img_size=config.img_size)

#converting coreset_as_dataset_fraction to coreset_size
if config.coreset_as_dataset_fraction is not None:
    config.coreset_size = int(len(train_dataset)*config.coreset_as_dataset_fraction)
    print('Coreset size is set to: ', config.coreset_size)
    
if len(train_dataset) <config.coreset_size:
    print('Coreset size is larger than the dataset size. Exiting....')
    sys.exit(0)

if config.coreset_method == 'random':
    coreset_inds = np.random.choice(len(train_dataset), config.coreset_size, replace=False)

elif  config.coreset_method == 'PGCS':
    print(config.coreset_method)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)
    
    pgcs = PGCS( imageEncoder = config.imageEncoder, projectionOperator=config.projectionOperator,m=config.prjection_hyper, partioningAlgo=config.partioningAlgo, K=config.partion_count)
    coreset_inds = pgcs.get_coreset(train_loader, config)

else:
    raise ValueError('Invalid coreset method')
print(coreset_inds)

# Save coreset_inds in a pickle file
index_file_path = os.path.join(config.index_folder, config.coreset_method, dataset_name)
if not os.path.exists(index_file_path):
    os.makedirs(index_file_path)

index_file = os.path.join(index_file_path, str(config.coreset_size)+'.pkl')
with open(index_file, 'wb') as f:
    pickle.dump(coreset_inds, f)

