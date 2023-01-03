import numpy as np 
import pandas as pd 

import os
import random

import torch 
import torch.nn.functional as F 
from torch import nn

import math
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder

from configuration import *

# def seed_setting():
#     seed = Config.SEED
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.use_deterministic_algorithms = True

def seed_setting(opt):
    seed = opt.SEED
    torch.cuda.set_device(opt.DEVICE)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
    
    
def stratify_df(opt, df):
    
    train_df = df.copy()
    
    train_df['fold'] = -1
    
    n_folds = Config.N_FOLDS
    
    strat_kfold = StratifiedKFold(n_splits=n_folds, random_state = opt.SEED, shuffle=True)

    for i, (_, train_index) in enumerate(strat_kfold.split(train_df.index, train_df['label_group'])):
        train_df.iloc[train_index,-1] = i

    train_df['fold'] = train_df['fold'].astype('int')
    
    if n_folds == 10:
        train = train_df[train_df.fold != 0].reset_index(drop=True)
        valid = train_df[train_df.fold == 0].reset_index(drop=True)
        
        return train,valid