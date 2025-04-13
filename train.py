import pandas as pd
import numpy as np
import os
import random
import re
import config as CFG
import wandb
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

df = pd.read_csv('./data/train.csv')

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

def train(model, fold, optimizer, train_loader, val_loader, scheduler, device):
    criterion = nn.MSELoss().to(device)
    #criterion = utils.WeightedMSELoss(weights)
    
    best_loss = float('inf')
        
    for epoch in tqdm(range(CFG.epochs)):
        # train
        model.train()
        train_loss = []
        for imgs, labels in iter(train_loader):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            pred = model(imgs)
            loss = criterion(pred, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        # validation
        model.eval()
        val_loss = []
        preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels in iter(val_loader):
                imgs = imgs.float().to(device)
                labels = labels.to(device)
                
                pred = model(imgs)
                loss = criterion(pred, labels)                
                
                preds.append(pred.cpu())
                all_labels.append(labels.cpu())
                val_loss.append(loss.item())
        
        # 하나의 큰 텐서로 변환
        preds = torch.cat(preds, dim=0).numpy()
        all_labels = torch.cat(all_labels, dim=0).numpy()

        # 행별 상관관계 계산
        cell_corr = [np.corrcoef(preds[i, :], all_labels[i, :])[0, 1] for i in range(preds.shape[0])]

        # 열별 상관관계 계산
        gene_corr = [np.corrcoef(preds[:, j], all_labels[:, j])[0, 1] for j in range(preds.shape[1])]

        _train_loss = np.mean(train_loss)
        _val_loss = np.mean(val_loss)
        val_metric = (np.mean(cell_corr) + np.max(gene_corr) + np.mean(gene_corr) * 2) / 4
         
        wandb.log({
            f"fold_{fold+1}_train_loss": np.mean(train_loss), 
            f"fold_{fold+1}_val_loss": np.mean(val_loss),
            f"fold_{fold+1}_val_meancorr_cells": np.mean(cell_corr),
            f"fold_{fold+1}_val_maxcorr_genes": np.max(gene_corr),
            f"fold_{fold+1}_val_meancorr_genes": np.mean(gene_corr),
            f"fold_{fold+1}_val_metric": val_metric
        })
       
        if scheduler is not None:
            scheduler.step(_val_loss)
            
        if best_loss > _val_loss:
            best_loss = _val_loss
            best_model = model
            best_epoch = epoch
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > CFG.early_stop_count:
                break
            
    return best_model, best_loss, best_epoch

NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)

wandb_name = CFG.model_name
if CFG.augment == True:
    wandb_name += "_aug"

wandb.init(project='MAI', name=wandb_name, group=CFG.model_name)

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"Fold {fold+1}")

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_label_vec = train_df.iloc[:,2:].values.astype(np.float32)
    val_label_vec = val_df.iloc[:,2:].values.astype(np.float32)

    train_dataset = MAIDataset(train_df['path'].values, train_label_vec, augment=CFG.augment)
    train_loader = DataLoader(train_dataset, batch_size = CFG.batch_size, shuffle=True, num_workers=0)    

    val_dataset = MAIDataset(val_df['path'].values, val_label_vec)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=0)
    
    model = BaseModel(CFG.model_name)
    model.to(device)
        
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG.learning_rate)
    
    if CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold_mode='abs', min_lr=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=CFG.min_lr)
    else:
        raise ValueError(f"Unknown scheduler name: {CFG.scheduler}.")
    
    best_model, best_loss, best_epoch = train(model, fold, optimizer, train_loader, val_loader, scheduler, device)
    
    model_save_path = f"./models/{wandb_name}_fold{fold+1}_epoch{best_epoch+1}_{best_loss:.5f}.pth"
    torch.save(best_model.state_dict(), model_save_path)

wandb.finish()