import numpy as np 
import pandas as pd 
import math
import random 
import os 
import cv2
import timm
import time
import argparse
import tqdm 
import h5py

import requests
import cv2

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

import torch 
from torch.utils.data import Dataset 
from torch import nn
import torch.nn.functional as F 

from annoy import AnnoyIndex    

from configuration import *
from transform import *
from utils import *
from activation import *
from model import *

from flask import request, jsonify, Blueprint
from flask import Flask


def get_encoded_groups(df):
    df_ = df.groupby(['label_group'])['image_name'].apply(lambda x: ' '.join(x))
    encoded_df = pd.DataFrame({'label_group':df_.index,
                             'image_group':df_.values})
    
    encoded_df['len'] = encoded_df.image_group.apply(lambda x : len(x.split(' '))) 
    #encoded_df["image_group"] = [x.split(',') for x in encoded_df['image_group']]
    return encoded_df

def load_ANN(opt):
    vector_size = opt.FC_DIM
    load_index = AnnoyIndex(vector_size, 'dot')
    load_index.load('embeddings/test.annoy')
    print("---------- Annoy Model Loading Successful ----------")
    return load_index

class serviceDataset():
    def __init__(self, img_url, transforms=None):

        self.img_url = img_url
        self.augmentations = transforms

    def query2emb(self):
        image_nparray = np.asarray(bytearray(requests.get(self.img_url).content), dtype=np.uint8)
        image = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']       
        image = image.unsqueeze(0)
        return image, torch.tensor(1)


def generate_embedding(opt, param):
    model = KfashionModel(opt)
    model.eval()
    model.load_state_dict(torch.load(opt.MODEL_PATH, map_location=opt.DEVICE))
    model = model.to(opt.DEVICE)
    
    
    datloader = serviceDataset(param, transforms=get_test_transforms())
    img, label = datloader.query2emb()
    
    with torch.no_grad():
        img = img.to(opt.DEVICE)
        label = torch.tensor(1).to(opt.DEVICE)
        feat = model(img,label)
        image_embedding = feat.detach().cpu().numpy()
    
    del model
    return image_embedding

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL_PATH', default='./weight/best.pt', help="Trained Weights path") # trained weight
    parser.add_argument('--DATA_DIR', default='/mnt/hdd1/wearly/compatibility_rec/data/images/', help="Image dataset path") # Local img DB
    parser.add_argument('--SEED', type=int, default=225)
    parser.add_argument('--DEVICE', default='cpu', help="cuda 0, 1 or cpu")
    parser.add_argument('--FC_DIM', type=int, default=512)
    parser.add_argument('--CLASSES', type=int, default=352, help="Number of group classes")
    parser.add_argument('--MODEL_NAME', default="tf_efficientnet_b4", help="timm model name")
    parser.add_argument('--TYP', default="test", help="train or test")
    parser.add_argument('--KNUM', type=int, default=5, help="ANN/KNN neighbors")
    opt = parser.parse_args()
    try:
        seed_setting(opt)
    except:
        pass
    
    #load_DBemb()
    start = time.time()
    load_index = load_ANN(opt)
    param = "https://cafe24img.poxo.com/dabainsang/web/product/big/20200203/a8fd1a8df656578aec043bb18fb1c0fb.jpg"
    query_emb = generate_embedding(opt, param)
    result = load_index.get_nns_by_vector(query_emb[0], opt.KNUM)
    print(f"{time.time()-start:.4f} sec")
    print(result)
