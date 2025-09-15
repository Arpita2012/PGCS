import os
import torch
import numpy as np
import logging
import time
import random
import pandas as pd
import sys
import json
import pickle

folder_path = "./src" # os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if folder_path not in sys.path:
    sys.path.insert(0, folder_path)


from iqa_config import set_config, parse_args,setup_seed
import iqa_architecture.maniqa 
from iqa_dataset import get_dataset
from iqa_train_test_utils import train_model, eval_model, testing_model

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset 
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter 

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


def set_logging(config):
    if not os.path.exists(config.log_path): 
        os.makedirs(config.log_path)
    filename = os.path.join(config.log_path, config.log_file)
    logging.basicConfig(
        level=logging.INFO,
        filename=filename,
        filemode='w',
        format='[%(asctime)s %(levelname)-8s] %(message)s',
        datefmt='%Y%m%d %H:%M:%S'
    )


cpu_num = 1
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

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
######################################################


if args.coreset_as_dataset_fraction is not None or config.coreset_as_dataset_fraction is not None:
    config.model_name= config.model_name + '_' + config.coreset_method + '_' + str(config.coreset_as_dataset_fraction)
else:
    config.model_name= config.model_name + '_' + config.coreset_method + '_' + str(config.coreset_size)


config.log_file = config.model_name + ".log"
config.tensorboard_path = os.path.join(config.tensorboard_path, config.dataset_name)
config.tensorboard_path = os.path.join(config.tensorboard_path, config.model_name)

config.ckpt_path = os.path.join(config.ckpt_path, config.dataset_name)
config.ckpt_path = os.path.join(config.ckpt_path, config.model_name)

config.log_path = os.path.join(config.log_path, config.dataset_name)

if not os.path.exists(config.ckpt_path):
    os.makedirs(config.ckpt_path)

if not os.path.exists(config.tensorboard_path):
    os.makedirs(config.tensorboard_path)

set_logging(config)
logging.info(config)

config.summary_file_path = os.path.join(config.log_path, 'summary.csv')

writer = SummaryWriter(config.tensorboard_path)


datsets={}
with open(config.dataset_meta_file, 'r') as f:
    dataset_meta = json.load(f)

dataset_info = dataset_meta[config.dataset_name]


type_l=['train',   'val' , 'test']
img = [224, 224, 224]
i=0
for type in type_l:
    print(img[i])
    dataset = get_dataset(config.dataset_name, dataset_info, type, img_size=img[i])
    loader = DataLoader(dataset=dataset, batch_size=config.batch_size, num_workers=config.num_workers,  shuffle=False)
    data = next(iter(loader))
    X = data['d_img_org'].numpy()
    y = data['score'].numpy()
    print(X.shape)
    print(y.shape)
    datsets[type] = dataset
    i=i+1



    logging.info('number of ' + str(type) + ' scenes: ' + str( len(dataset)))
    

train_dataset = datsets['train']
val_dataset = datsets['val']
test_dataset = datsets['test']


#converting coreset_as_dataset_fraction to coreset_size
if config.coreset_as_dataset_fraction is not None:
    config.coreset_size = int(len(train_dataset)*config.coreset_as_dataset_fraction)
    print('Coreset size is set to: ', config.coreset_size)

if config.coreset_size < 5 and config.coreset_method != 'FULL' and config.coreset_method != None:
    logging.info('Coreset size is too small. Exiting....')
    sys.exit(0)

if config.coreset_size >len(train_dataset):
    logging.info('Coreset size is larger than the train dataset size. Exiting....')
    sys.exit(0)


#########################################################################################################
#set dataloader for train_dataset based on coreset_method
if config.coreset_method == None or config.coreset_method == 'FULL':
    #No method specified train on full dataset
    config.coreset_method = 'FULL'
    config.coreset_size = len(train_dataset)
    config.coreset_as_dataset_fraction = 1.0
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, num_workers=config.num_workers,  shuffle=True)
    
else:
    #Load coreset indices
    
    index_file_path = os.path.join(config.index_folder, config.coreset_method, config.dataset_name)
    index_file = os.path.join(index_file_path, str(config.coreset_size)+'.pkl')
    with open(index_file, 'rb') as f:
        coreset_inds = pickle.load(f)
    coreset_subset = Subset(train_dataset, coreset_inds)
    train_loader = DataLoader(dataset=coreset_subset, batch_size=config.batch_size, num_workers=config.num_workers,  shuffle=True)
    logging.info('number of '+ config.coreset_method+' coreset scenes: ' + str( len(coreset_subset)))
    


val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
print("VAL LOADER IMAGE")
data = next(iter(val_loader))
X = data['d_img_org'].numpy()
y = data['score'].numpy()
print(X.shape)
print(y.shape)



#########################################################################################################

device = 'cuda' if torch.cuda.is_available() else 'cpu'

net = iqa_architecture.maniqa.MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp, patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size, depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

# print(net)

logging.info('{} : {} [M]'.format('#Params', sum(map(lambda x: x.numel(), net.parameters())) / 10 ** 6))

net = nn.DataParallel(net)
net = net.to(device) 


# loss function
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(
    net.parameters(),
    lr=config.learning_rate,
    weight_decay=config.weight_decay,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.T_max, eta_min=config.eta_min)

# train & validation
losses, scores = [], []
best_srocc = 0
best_plcc = 0
main_score = 0

early_stop_cnt=0;

for epoch in range(0, config.n_epoch):
    start_time = time.time()
    logging.info('Running training epoch {}'.format(epoch + 1))
    loss_train, rho_s_train, rho_p_train = train_model(epoch, net, criterion, optimizer, scheduler, train_loader)

    writer.add_scalar("Train_loss", loss_train, epoch)
    writer.add_scalar("SRCC", rho_s_train, epoch)
    writer.add_scalar("PLCC", rho_p_train, epoch)



    if (epoch + 1) % config.val_freq == 0:
        logging.info('Starting eval...')
        logging.info('Running validation in epoch {}'.format(epoch + 1))
        loss_val, rho_s_val, rho_p_val = eval_model(config, epoch, net, criterion, val_loader)
        logging.info('Eval done...')

        if rho_s_val + rho_p_val > main_score:

            best_train_srocc = rho_s_train
            best_train_plcc = rho_p_train
            best_train_loss = loss_train

            early_stop_cnt=0
            main_score = rho_s_val + rho_p_val
            best_srocc_val = rho_s_val
            best_plcc_val = rho_p_val

            logging.info('======================================================================================')
            logging.info('============================== best main score is {} ================================='.format(main_score))
            logging.info('======================================================================================')

            # save weights
            model_name = "epoch{}.pt".format(epoch + 1)
            model_save_path = os.path.join(config.ckpt_path, model_name)
            # torch.save(net.module.state_dict(), model_save_path) # save model commented out to save space on server| only save one best one , not all the best ones

            best_model_save_path = os.path.join(config.ckpt_path, "best.pt")
            torch.save(net.module.state_dict(), best_model_save_path)


            logging.info('Saving weights and model of epoch{}, SRCC:{}, PLCC:{}'.format(epoch + 1, best_srocc_val, best_plcc_val))
        else:
            early_stop_cnt += 1
            if early_stop_cnt == config.early_stop:
                logging.info('Early stopping at epoch {}'.format(epoch + 1))
                break
    logging.info('Epoch {} done. Time: {:.2}min'.format(epoch + 1, (time.time() - start_time) / 60))

logging.info('Training done...')
logging.info('======================================================================================')
logging.info('============================== best main score is {} ================================='.format(main_score))
logging.info('======================================================================================')


#### testing
logging.info('Testing...')
logging.info('Loading best model...')
net = iqa_architecture.maniqa.MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp, patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size, depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)



net.load_state_dict(torch.load(best_model_save_path))

net = nn.DataParallel(net)
net = net.to(device) 


net.eval()
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
loss_test, rho_s_test, rho_p_test = testing_model(config,  net, criterion, test_loader)
logging.info('Testing done...')
logging.info('LOSS: {}, SRCC: {}, PLCC: {}'.format(loss_test, rho_s_test, rho_p_test))
logging.info('======================================================================================')


import time
cuurent_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

test_row = {'time_stamp': cuurent_time ,'type': 'test', 'dataset_name': config.dataset_name, 'coreset_method': config.coreset_method, 'coreset_size': config.coreset_size, 'coreset_as_dataset_fraction':config.coreset_as_dataset_fraction, 'loss': loss_test, 'SRCC': rho_s_test, 'PLCC': rho_p_test}

val_row = {'time_stamp': cuurent_time ,'type': 'val', 'dataset_name': config.dataset_name, 'coreset_method': config.coreset_method, 'coreset_size': config.coreset_size, 'coreset_as_dataset_fraction':config.coreset_as_dataset_fraction, 'loss': loss_val, 'SRCC': best_srocc_val, 'PLCC': best_plcc_val}

train_row = {'time_stamp': cuurent_time , 'type': 'train', 'dataset_name': config.dataset_name, 'coreset_method': config.coreset_method, 'coreset_size': config.coreset_size, 'coreset_as_dataset_fraction':config.coreset_as_dataset_fraction, 'loss': best_train_loss, 'SRCC': best_train_srocc, 'PLCC': best_train_plcc}

if os.path.exists(config.summary_file_path):
    df = pd.read_csv(config.summary_file_path)
else:
    df = pd.DataFrame(columns=['time_stamp', 'type', 'dataset_name', 'coreset_method', 'coreset_size', 'loss', 'SRCC', 'PLCC'])


df = pd.concat([df, pd.DataFrame([train_row, val_row, test_row])], ignore_index=True)

df.to_csv(config.summary_file_path, index=False)


df.to_csv(config.summary_file_path,  header=True, index=False)
print(df)





