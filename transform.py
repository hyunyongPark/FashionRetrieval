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

from configuration import *

def get_train_transforms():
    return albumentations.Compose(
        [
            albumentations.Resize(Config.IMG_SIZE, Config.IMG_SIZE, always_apply=True),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            albumentations.Rotate(limit=120, p=0.8),
            albumentations.RandomBrightness(limit=(0.09, 0.6), p=0.5),
            albumentations.Normalize(mean = Config.MEAN, std = Config.STD),
            ToTensorV2(p=1.0),
        ])


def get_valid_transforms():

    return albumentations.Compose(
        [
            albumentations.Resize(Config.IMG_SIZE, Config.IMG_SIZE,always_apply=True),
            albumentations.Normalize(),
        ToTensorV2(p=1.0)
        ]
    )



def get_test_transforms():
    
    return albumentations.Compose(
        [
            albumentations.Resize(Config.IMG_SIZE, Config.IMG_SIZE,always_apply=True),
            albumentations.Normalize(),
        ToTensorV2(p=1.0)
        ]
    )
