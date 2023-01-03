import os

class Config:
    IMG_SIZE = 512
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    N_FOLDS = 10
    
    SCALE = 30 
    MARGIN = 0.5
    
    weight_decay = 0.0
    optimizer_name = 'adam'