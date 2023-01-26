import numpy as np 
import pandas as pd 
import time

import math
import random 
import os 
import cv2
import timm
import argparse

import tqdm 
import h5py

import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2

import torch 
from torch.utils.data import Dataset 
from torch import nn
import torch.nn.functional as F 

import plotly.express as px

import gc
from annoy import AnnoyIndex    

from configuration import *
from transform import *
from utils import *
from dataset import *


def get_encoded_groups(df):
    df_ = df.groupby(['label_group'])['image_name'].apply(lambda x: ' '.join(x))
    encoded_df = pd.DataFrame({'label_group':df_.index,
                             'image_group':df_.values})
    
    encoded_df['len'] = encoded_df.image_group.apply(lambda x : len(x.split(' '))) 
    #encoded_df["image_group"] = [x.split(',') for x in encoded_df['image_group']]
    return encoded_df

def gain_intersections(df):
    y_true = df['label_group']
    y_pred = df['pred_matches']
    
    start = time.time()
    y_true = y_true.apply(lambda x: frozenset(x.split()))
    y_pred = y_pred.apply(lambda x: frozenset(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in tqdm.tqdm(zip(y_true, y_pred))])
    print(f"{time.time()-start:.4f} sec") 
    return intersection

# https://www.kaggle.com/code/tanulsingh077/metric-learning-image-tfidf-inference
def get_neighbors(df, embeddings, KNN = 30):
    model = NearestNeighbors(n_neighbors = KNN)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    
    predictions = []
    for k in tqdm(range(embeddings.shape[0])):
        idx = np.where(distances[k,] < 2.7)[0]
        ids = indices[k,idx]
        posting_ids = df['image_name'].iloc[ids].values
        predictions.append(posting_ids)
        
    del model, distances, indices
    gc.collect()
    return df, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', default='/mnt/hdd1/wearly/compatibility_rec/data/images/', help="Image dataset path")
    parser.add_argument('--TRAIN_CSV', default='data/separ_train.csv', help="Train dataset path")
    parser.add_argument('--TEST_CSV', default='data/separ_test.csv', help="Test dataset path")
    parser.add_argument('--SEED', type=int, default=225)
    parser.add_argument('--DEVICE', default='cuda:1', help="cuda 0, 1 or cpu")
    parser.add_argument('--FC_DIM', type=int, default=512)
    parser.add_argument('--KNUM', type=int, default=50, help="ANN/KNN neighbors")
    opt = parser.parse_args()
    
    try:
        seed_setting(opt)
    except:
        pass
    
    h5f = h5py.File('embeddings/all_emb.h5','r')
    print('\033[94m' + f'----Embedding Loading Successful----' + '\033[0m')
    image_embeddings = h5f['all_data'][:]

    # annoy indexing
    vector_size = opt.FC_DIM
    load_index = AnnoyIndex(vector_size, 'dot')
    load_index.load('embeddings/test.annoy')
    print('\033[94m' + f'----Annoy Model Loading Successful----' + '\033[0m')
    
    start = time.time()
    df = pd.read_pickle("embeddings/all_dt_eval.pkl")
    df.pred_matches = df.pred_matches.astype("category")
    df.label_group = df.label_group.astype("category")
    print(f"{time.time()-start:.4f} sec")
    print('\033[94m' + f'----testset pkl Loading Successful----' + '\033[0m')
    
    te = pd.read_csv("./separ_test.csv", index_col=0)
    te = te[["image_name"]]
    df = pd.merge(te, df, on="image_name")
    
    intersection = gain_intersections(df)
    df["intersection"] = intersection
    df["precision"] = df["intersection"] / opt.KNUM
    score_map = df["precision"].mean()
    print('\033[96m' + f'\tmAP@{opt.KNUM}:{score_map}' + '\033[0m')
    
    