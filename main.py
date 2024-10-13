import pandas as pd
import numpy as np
import os
import random
import config as CFG
import wandb

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold

from dataset import MAIDataset
from models import SwinV2_s, SwinV2_t

import warnings
warnings.filterwarnings(action='ignore') 

df = pd.read_csv('./data/train.csv')
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

def train(model, fold, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.MSELoss().to(device)
    
    best_loss = 99999999
    
    for epoch in CFG.epochs:
        # train
        model.train()
        train_loss = []
        for imgs, labels in iter(train_loader):
            imgs = imgs.float().to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            output = model(imgs)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss.append(loss.item())
                    
        # validation
        model.eval()
        val_loss = []
        with torch.no_grad():
            for imgs, labels in iter(val_loader):
                imgs = imgs.float().to(device)
                labels = labels.to(device)
            
                pred = model(imgs)
                loss = criterion(pred, labels)
            
                val_loss.append(loss.item())
                
        _train_loss = np.mean(train_loss)
        _val_loss = np.mean(val_loss)
        wandb.log({f"fold_{fold+1}_train_loss": _train_loss, f"fold_{fold+1}_val_loss": _val_loss})
       
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

def inference(model, test_loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for imgs in test_loader:
            imgs = imgs.to(device).float()
            pred = model(imgs)
            
            preds.append(pred.detach().cpu())
    
    preds = torch.cat(preds).numpy()

    return preds

test_dataset = MAIDataset(test_df['path'].values, None, transform='test')
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=0)

preds = np.zeros((len(test_df), CFG.gene_size))
loss = []

NFOLD = 5
kf = KFold(n_splits=NFOLD, shuffle=True, random_state=42)

wandb.init(project='MAI', name=CFG.exp_name, group=CFG.model_name, config=CFG, reinit=True)

for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
    print(f"Fold {fold+1}")

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    
    train_label_vec = train_df.iloc[:,2:].values.astype(np.float32)
    val_label_vec = val_df.iloc[:,2:].values.astype(np.float32)

    train_dataset = MAIDataset(train_df['path'].values, train_label_vec, augment=True)
    train_loader = DataLoader(train_dataset, batch_size = CFG.batch_size, shuffle=True, num_workers=0)    

    val_dataset = MAIDataset(val_df['path'].values, val_label_vec)
    val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=0)

    if CFG.model_name == 'swinv2_s':
        model = SwinV2_s()
    elif CFG.model_name == 'swinv2_t':
        model = SwinV2_t()
    else:
        raise ValueError(f"Unknown model name: {CFG.model_name}.")
        
    model.eval()
    optimizer = torch.optim.Adam(params = model.parameters(), lr = CFG.learning_rate)
    
    if CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, threshold_mode='abs', min_lr=CFG.min_lr)
    elif CFG.scheduler == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=CFG.min_lr)
    else:
        raise ValueError(f"Unknown scheduler name: {CFG.scheduler}.")

    best_model, best_loss, best_epoch = train(model, fold, optimizer, train_loader, val_loader, scheduler, device)
    
    model_save_path = f"./models/{CFG.model_name}_{CFG.exp_name}_fold{fold+1}_epoch{best_epoch+1}_loss_{best_loss:.5f}.pth"
    torch.save(best_model.state_dict(), model_save_path)
    loss.append(best_loss)
    
    preds += inference(best_model, test_loader, device) / NFOLD

wandb.finish()
        
submit = pd.read_csv('./data/sample_submission.csv')
submit.iloc[:, 1:] = np.array(preds).astype(np.float32)
submit.to_csv(f'./submission/{CFG.model_name}_{CFG.exp_name}_{np.mean(loss):.5f}.csv', index=False)