import numpy as np 
import pandas as pd 
import argparse
import math
import random 
import os 
import cv2
import timm

from tqdm import tqdm 
import h5py

import torch 
from torch.utils.data import Dataset 
from torch import nn
import torch.nn.functional as F 

from annoy import AnnoyIndex

from configuration import *
from transform import *
from activation import *
from model import *
from utils import *
from dataset import *    


def get_encoded_groups(df):
    df_ = df.groupby(['label_group'])['image_name'].apply(lambda x: ','.join(x))
    encoded_df = pd.DataFrame({'label_group':df_.index,
                             'image_group':df_.values})
    
    encoded_df['len'] = encoded_df.image_group.apply(lambda x : len(x.split(','))) 
    encoded_df["image_group"] = [x.split(',') for x in encoded_df['image_group']]
    return encoded_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL_PATH', default='weight/best.pt', help="Trained Weights path")
    parser.add_argument('--DATA_DIR', default='/mnt/hdd1/wearly/compatibility_rec/data/images/', help="Image dataset path")
    parser.add_argument('--TRAIN_CSV', default='data/separ_train.csv', help="Train dataset path")
    parser.add_argument('--TEST_CSV', default='data/separ_test.csv', help="Test dataset path")
    parser.add_argument('--SEED', type=int, default=225)
    parser.add_argument('--DEVICE', default='cuda:1', help="cuda 0, 1 or cpu")
    opt = parser.parse_args()
    
    try:
        seed_setting(opt)
    except:
        pass
    tr_df = pd.read_csv(opt.TRAIN_CSV,index_col=0)
    te_df = pd.read_csv(opt.TEST_CSV,index_col=0)
    df = tr_df.append(te_df).reset_index(drop=True)
    image_paths = opt.DATA_DIR + df['image_name']

    encoded_df = get_encoded_groups(df)
    encoded_dict = encoded_df[['label_group', 'image_group']].to_dict('records')

    df["label_groups"] = [encoded_dict[df["label_group"][i]]["image_group"] for i in tqdm(range(len(df)))]
    
    h5f = h5py.File('./embeddings/all_emb.h5','r')
    image_embeddings = h5f['all_data'][:]
    print("---------- Embedding Loading Successful ----------")
    
    
    # annoy indexing
    vector_size = image_embeddings.shape[1]
    index = AnnoyIndex(vector_size, 'dot')
    data = []
    vectors = image_embeddings
    
    for idx in range(len(vectors)):
        data.append({'idx':idx, 'image_name':df.image_name[idx], 'vector':vectors[idx], 'label_groups':df.label_groups[idx]})
        index.add_item(idx, vectors[idx])

    index.build(70)
    index.save('./embeddings/test.annoy')










