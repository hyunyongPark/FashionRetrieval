import numpy as np 
import pandas as pd 

import argparse
import os 
import cv2 

import torch 
import torch.nn.functional as F 
from torch import nn 
from torch.optim import Adam
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import math
import neptune
from sklearn.model_selection import StratifiedKFold

import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score

from torch.optim.lr_scheduler import StepLR, ExponentialLR, OneCycleLR, _LRScheduler, ReduceLROnPlateau

from configuration import *
from transform import *
from activation import *
from model import *
from utils import *
from dataset import *
#from sklearn.metrics import f1_score

def train_fn(model, data_loader, optimizer, scheduler, i):
    model.train()
    fin_loss = 0.0
    tk = tqdm.tqdm(data_loader, desc = "Epoch" + " [TRAIN] " + str(i+1))

    for t,data in enumerate(tk):
        for k,v in data.items():
            data[k] = v.to(opt.DEVICE)
        optimizer.zero_grad()
        _, loss = model(**data)
        loss.mean().backward()
        optimizer.step() 
        fin_loss += loss.mean().item() 

        tk.set_postfix({'loss' : '%.6f' %float(fin_loss/(t+1)), 'LR' : optimizer.param_groups[0]['lr']})
        
        #neptune.log_metric('loss_tr',float(fin_loss/(t+1)))
        #neptune.log_metric('train_lr',optimizer.param_groups[0]['lr'])
        
    #scheduler.step()

    return fin_loss / len(data_loader)

def eval_fn(model, data_loader, i):
    model.eval()
    fin_loss = 0.0
    tk = tqdm.tqdm(data_loader, desc = "Epoch" + " [VALID] " + str(i+1))

    with torch.no_grad():
        for t,data in enumerate(tk):
            for k,v in data.items():
                data[k] = v.to(opt.DEVICE)
            _, loss = model(**data)
            fin_loss += loss.mean().item() 

            tk.set_postfix({'loss' : '%.6f' %float(fin_loss/(t+1))})
            
            #neptune.log_metric('loss_valid', float(fin_loss/(t+1)))
            
        return fin_loss / len(data_loader)
    
    
def run_training(opt):
    
    seed_setting(opt)
    
    df = pd.read_csv(opt.TRAIN_CSV, index_col=0)
    df = df.reset_index(drop=True)
    
    train,valid = stratify_df(opt, df)
    print(f"train shape : {train.shape}")
    print(f"validation shape : {valid.shape}")
    
    print(train.label_group.nunique())
    print(valid.label_group.nunique())
    
    #train
    train_dataset = KfashionDataset(opt, train, transform = get_train_transforms())    
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = opt.BATCH_SIZE,
        pin_memory = True,
        num_workers = opt.NUM_WORKERS,
        shuffle = True,
        drop_last = True
    )
    
    #valid 추가
    valid_dataset = KfashionDataset(opt, valid, transform = get_valid_transforms())
    validloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size = opt.BATCH_SIZE,
        num_workers = opt.NUM_WORKERS,
        shuffle = False,
        pin_memory = True,
        drop_last = False
        )
    
    
    model = KfashionModel(opt)
    #model = nn.DataParallel(model)
    model.to(opt.DEVICE)
    
    #existing_layer = torch.nn.SiLU
    #new_layer = Mish()
    #model = replace_activations(model, existing_layer, new_layer) # in eca_nfnet_l0 SiLU() is used, but it will be replace by Mish()
    
    optimizer = Adam(model.parameters(), lr = opt.LR_START)
    #scheduler = OneCycleLR(optimizer, max_lr = 2e-3, steps_per_epoch = len(trainloader),epochs=Config.EPOCHS)
    
    save_dir = f'./{opt.MODEL_NAME}_{opt.EPOCHS}_{Config.optimizer_name}'
    
    if os.path.exists(save_dir) == False :
        print('Making Weights Folder')
        os.mkdir(save_dir)
    
    print("=======Start learning=======")
    best_valid_score = 1000
    for i in range(opt.EPOCHS):
        avg_loss_train = train_fn(model, trainloader, optimizer, None, i)
        avg_loss_valid = eval_fn(model, validloader,i)
        if avg_loss_valid <= best_valid_score:
            torch.save(model.state_dict(),f'{save_dir}/best_{i}EpochStep.pt')
            best_valid_score = avg_loss_valid

            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', default='/mnt/hdd1/wearly/compatibility_rec/data/images/', help="Image dataset path")
    parser.add_argument('--TRAIN_CSV', default='./data/separ_train.csv', help="Train dataset path")
    parser.add_argument('--SEED', type=int, default=225)
    parser.add_argument('--EPOCHS', type=int, default=50)
    parser.add_argument('--BATCH_SIZE', type=int, default=16, help='Total batch size for all GPUs')
    parser.add_argument('--DEVICE', default='cuda:1', help="cuda 0, 1 or cpu")
    parser.add_argument('--NUM_WORKERS', type=int, default=4)
    parser.add_argument('--CLASSES', type=int, default=352, help="Number of group classes")
    parser.add_argument('--LR_START', type=float, default=1e-5)
    parser.add_argument('--FC_DIM', type=int, default=512)
    parser.add_argument('--MODEL_NAME', default="tf_efficientnet_b4", help="timm model name")
    parser.add_argument('--TYP', default="train", help="train or test")
    opt = parser.parse_args()

    run_training(opt)
