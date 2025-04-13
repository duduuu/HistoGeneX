import pandas as pd
import numpy as np
import os
import random
import config as CFG
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from dataset import MAIDataset
from models import BaseModel
import utils

import warnings
warnings.filterwarnings(action='ignore') 

import importlib
importlib.reload(CFG)

test_df = pd.read_csv('./data/test.csv')

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(CFG.seed) # Seed 고정

def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in tqdm(test_loader):
            imgs = imgs.to(device).float()
            pred = model(imgs)
            
            preds.append(pred.detach().cpu())
    
    preds = torch.cat(preds).numpy()

    return preds

test_dataset = MAIDataset(test_df['path'].values, None)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=0)

preds = np.zeros((len(test_df), CFG.gene_size))

NFOLD = 5

wandb_name = "eva02_s_aug"

for fold in range(NFOLD):
    print(f"Fold {fold+1}")
    
    model = BaseModel(CFG.model_name)
    model = model.to(device)
    
    model_path = utils.get_model_path("models", wandb_name, fold + 1)
    model.load_state_dict(torch.load(model_path))
    print(model_path)
    
    preds += inference(model, test_loader, device) / NFOLD

submit = pd.read_csv('./data/sample_submission.csv')
submit.iloc[:, 1:] = np.array(preds).astype(np.float32)
submit.to_csv(f'./submission/{wandb_name}_ver2.csv', index=False)