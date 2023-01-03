import numpy as np 
import pandas as pd 

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
    
# https://data-newbie.tistory.com/472
# 데이터 크기 확인 함수
def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)

## 타입별 평균 크기 확인 함수
def type_memory(data) :
    for dtype in ['float','int','object']:
        selected_dtype = data.select_dtypes(include=[dtype])
        mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
        mean_usage_mb = mean_usage_b / 1024 ** 2
        print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))

## 이산형 데이터 사이즈 축소 함소
def int_memory_reduce(data) :
    data_int = data.select_dtypes(include=['int'])
    converted_int = data_int.apply(pd.to_numeric,downcast='unsigned')
    print(f"Before : {mem_usage(data_int)} -> After : {mem_usage(converted_int)}")
    data[converted_int.columns] = converted_int
    return data

## 연속형 데이터 사이즈 축소 함소
def float_memory_reduce(data) :
    data_float = data.select_dtypes(include=['float'])
    converted_float = data_float.apply(pd.to_numeric,downcast='float')
    print(f"Before : {mem_usage(data_float)} -> After : {mem_usage(converted_float)}")
    data[converted_float.columns] = converted_float
    return data

## 문자형 데이터 사이즈 축소 함소
def object_memory_reduce(data) :
    gl_obj = data.select_dtypes(include=['object']).copy()
    converted_obj = pd.DataFrame()
    for col in gl_obj.columns:
        num_unique_values = len(gl_obj[col].unique())
        num_total_values = len(gl_obj[col])
        if num_unique_values / num_total_values < 0.5:
            converted_obj.loc[:,col] = gl_obj[col].astype('category')
        else:
            converted_obj.loc[:,col] = gl_obj[col]
    print(f"Before : {mem_usage(gl_obj)} -> After : {mem_usage(converted_obj)}")
    data[converted_obj.columns] = converted_obj
    return data

def get_encoded_groups(df):
    df_ = df.groupby(['label_group'])['image_name'].apply(lambda x: ' '.join(x))
    encoded_df = pd.DataFrame({'label_group':df_.index,
                             'image_group':df_.values})
    
    encoded_df['len'] = encoded_df.image_group.apply(lambda x : len(x.split(' '))) 
    #encoded_df["image_group"] = [x.split(',') for x in encoded_df['image_group']]
    return encoded_df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', default='/mnt/hdd1/wearly/compatibility_rec/data/images/', help="Image dataset path")
    parser.add_argument('--TRAIN_CSV', default='./data/separ_train.csv', help="Train dataset path")
    parser.add_argument('--TEST_CSV', default='./data/separ_test.csv', help="Test dataset path")
    parser.add_argument('--SEED', type=int, default=225)
    parser.add_argument('--DEVICE', default='cuda:1', help="cuda 0, 1 or cpu")
    parser.add_argument('--FC_DIM', type=int, default=512)
    parser.add_argument('--KNUM', type=int, default=10, help="ANN/KNN neighbors")
    opt = parser.parse_args()
    
    try:
        seed_setting(opt)
    except:
        pass
    tr_df = pd.read_csv(opt.TRAIN_CSV,index_col=0)
    te_df = pd.read_csv(opt.TEST_CSV,index_col=0)
    df = tr_df.append(te_df).reset_index(drop=True)
    df = df[["image_name", "label_group"]]
    #df.to_csv("separ_all.csv", index=0)
    #df = pd.read_csv("separ_all.csv")
    image_paths = opt.DATA_DIR + df['image_name']
    
    encoded_df = get_encoded_groups(df)
    encoded_dict = encoded_df[['label_group', 'image_group']].to_dict('records')

    df["label_group"] = [encoded_dict[i]["image_group"] for i in df.label_group.values.tolist()]
    #df = object_memory_reduce(df)
    
    h5f = h5py.File('embeddings/all_emb.h5','r')
    print("---------- Embedding Loading Successful ----------")
    image_embeddings = h5f['all_data'][:]

    # annoy indexing
    vector_size = opt.FC_DIM
    load_index = AnnoyIndex(vector_size, 'dot')
    load_index.load('embeddings/test.annoy')
    print("---------- Annoy Model Loading Successful ----------")
    
    
    predictions = []

    for i in tqdm.tqdm(range(len(df))):
        result = load_index.get_nns_by_vector(image_embeddings[i], opt.KNUM)
        predict_list = [df.loc[df.index == r, "image_name"].values[0] for r in result]
        predictions.append(predict_list)

    df['pred_matches'] = predictions
    df['pred_matches'] = [' '.join(x) for x in tqdm.tqdm(df['pred_matches'].values.tolist())]
    
    #df = object_memory_reduce(df)
    df.to_pickle("embeddings/all_dt_eval.pkl")







