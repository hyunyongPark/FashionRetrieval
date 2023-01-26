import numpy as np 
import pandas as pd 

import os 
import cv2 

import albumentations 
from albumentations.pytorch.transforms import ToTensorV2

import torch 
import torch.nn.functional as F 
from torch import nn 

import math
from sklearn.model_selection import StratifiedKFold

from transform import *
from configuration import *

class KfashionDataset(torch.utils.data.Dataset):
    
    def __init__(self, opt, df, transform=None):
        self.df = df
        self.root_dir = opt.DATA_DIR
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        
        row = self.df.iloc[idx]
        img_path = os.path.join(self.root_dir, row.image_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = row.label_group
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        
        return {
            'image' : image,
            'label' : torch.tensor(label).long()
        }
    
    
class KfashionDataset_test(torch.utils.data.Dataset):
    def __init__(self, image_paths, transforms=None):

        self.image_paths = image_paths
        self.augmentations = transforms

    def __len__(self):
        return self.image_paths.shape[0]

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']       
    
        return image,torch.tensor(1)
    