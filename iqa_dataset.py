import torch
from src.utils.process import RandCrop, ToTensor, Normalize, five_point_crop
from src.utils.process import RandRotation, RandHorizontalFlip
import pandas as pd
import numpy as np
import cv2
import os
from torchvision import transforms

class Dataset_Parent(torch.utils.data.Dataset):
    def __init__(self, dis_path, txt_file_name, list_file_name, q_max, q_min,transform=None,img_size=224):
        super(Dataset_Parent, self).__init__()
        self.dis_path = dis_path
        self.txt_file_name = txt_file_name
        self.list_file_name = list_file_name
        self.img_size = img_size


        
        self.transform = transform

        self.q_max = q_max
        self.q_min = q_min

      
        # Read the CSV file
        data = pd.read_csv(self.list_file_name, sep=',')#,  names=['file', 'score'])

        # print(data)

        # Convert the score column to float
        data['score'] = data['score'].astype(float)

        # Get the dis_files_data and score_data
        dis_files_data = data['file'].tolist()
        score_data = data['score'].tolist()
        #########################################
        

        # reshape score_list (1xn -> nx1)
        score_data = np.array(score_data)
        score_data = self.normalization(score_data)
        score_data = list(score_data.astype('float').reshape(-1, 1))

        self.data_dict = {'d_img_list': dis_files_data, 'score_list': score_data}

        self.labels = np.squeeze( score_data);

    def normalization(self, data):
     
    
        range = self.q_max - self.q_min
        return (data - self.q_min) / range

    def __len__(self):
        return len(self.data_dict['d_img_list'])
    
    def __getitem__(self, idx):
        d_img_name = self.data_dict['d_img_list'][idx]
        img_full_path = os.path.join(self.dis_path, d_img_name)
        img_full_path = img_full_path.replace("\\", "/") #PIQ2023 dataset has backslashes in the path
        d_img = cv2.imread(img_full_path, cv2.IMREAD_COLOR)
        d_img = cv2.resize(d_img, (self.img_size, self.img_size), interpolation=cv2.INTER_CUBIC)
        d_img = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
        d_img = np.array(d_img).astype('float32') / 255
        d_img = np.transpose(d_img, (2, 0, 1))
        score = self.data_dict['score_list'][idx]

        sample = {
            'd_img_org': d_img,
            'score': score
        }
        if self.transform:
            sample = self.transform(sample)
        return sample


def get_dataset (dataset_name, dataset_info, type = 'train',img_size=224):
    
    file_type = ''
    if type == 'train':
        file_type = dataset_info['train_file']
    
    elif type == 'test':
        file_type = dataset_info['test_file'] 

    elif type == 'val':
        file_type = dataset_info['val_file'] 
    else:
        file_type = ""


    dataset = Dataset_Parent(
                dis_path=os.path.join(dataset_info['root'],dataset_info["img_dir"]), \
                txt_file_name=os.path.join( dataset_info['root'] , dataset_info['label']), \
                list_file_name=os.path.join( dataset_info['root'],file_type ), \
                # transform=transforms.Compose([RandCrop(patch_size=dataset_info["crop_size"]), \
                # Normalize(0.5, 0.5), RandHorizontalFlip(prob_aug=dataset_info["prob_aug"]), ToTensor()]),  \
                transform=transforms.Compose([\
                Normalize(0.5, 0.5), RandHorizontalFlip(prob_aug=dataset_info["prob_aug"]), ToTensor()]),  \
                q_max=dataset_info["q_max"], \
                q_min=dataset_info["q_min"], img_size = img_size)


    return dataset;