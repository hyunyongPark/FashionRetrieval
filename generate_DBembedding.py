import numpy as np 
import pandas as pd 
import argparse

import math
import random 
import os 
import cv2

from tqdm import tqdm 
import h5py

import torch 
from torch.utils.data import Dataset 
from torch import nn
import torch.nn.functional as F 

import gc

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



def get_image_embeddings(opt, image_paths):
    embeds = []
    
    model = KfashionModel(opt)
    model.to(opt.DEVICE)
    model.eval()
    
    if opt.MODEL_NAME == 'eca_nfnet_l0':
        model = replace_activations(model, torch.nn.SiLU, Mish())

    model.load_state_dict(torch.load(opt.MODEL_PATH, map_location=opt.DEVICE))
    model = model.to(opt.DEVICE)
    

    image_dataset = KfashionDataset_test(image_paths=image_paths,transforms=get_test_transforms())
    image_loader = torch.utils.data.DataLoader(
        image_dataset,
        batch_size=opt.BATCH_SIZE,
        #pin_memory=True, 
        drop_last=False,
        num_workers=opt.NUM_WORKERS
    )
    
    
    with torch.no_grad():
        for img,label in tqdm(image_loader): 
            img = img.to(opt.DEVICE)
            label = label.to(opt.DEVICE)
            feat = model(img,label)
            #print(feat[0].shape)
            image_embeddings = feat.detach().cpu().numpy()
            embeds.append(image_embeddings)
    
    del model
    image_embeddings = np.concatenate(embeds)
    print(f'Our image embeddings shape is {image_embeddings.shape}')
    del embeds
    gc.collect()
    return image_embeddings



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--MODEL_PATH', default='weight/best.pt', help="Trained Weights path")
    parser.add_argument('--DATA_DIR', default='/mnt/hdd1/wearly/compatibility_rec/data/images/', help="Image dataset path")
    parser.add_argument('--TRAIN_CSV', default='data/separ_train.csv', help="Train dataset path")
    parser.add_argument('--TEST_CSV', default='data/separ_test.csv', help="Test dataset path")
    parser.add_argument('--SEED', type=int, default=225)
    parser.add_argument('--BATCH_SIZE', type=int, default=16, help='Total batch size for all GPUs')
    parser.add_argument('--DEVICE', default='cuda:1', help="cuda 0, 1 or cpu")
    parser.add_argument('--NUM_WORKERS', type=int, default=4)
    parser.add_argument('--CLASSES', type=int, default=352, help="Number of group classes")
    parser.add_argument('--FC_DIM', type=int, default=512)
    parser.add_argument('--MODEL_NAME', default="tf_efficientnet_b4", help="timm model name")
    parser.add_argument('--TYP', default="test", help="train or test")
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
    
    image_embeddings = get_image_embeddings(opt, image_paths)
    
    
    save_dir = f'./embeddings'
    
    if os.path.exists(save_dir) == False :
        os.mkdir(save_dir)
    
    h5f = h5py.File(f'{save_dir}/all_emb.h5', 'w')
    h5f.create_dataset('all_data', data=image_embeddings)
    h5f.close()
    print("---------Embeddings saved complete---------")












