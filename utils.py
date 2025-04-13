import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr

# 음의 피어슨 상관계수 손실 함수 정의
def negative_pearson_loss(preds, targets):
    preds_mean = torch.mean(preds, dim=1, keepdim=True)
    targets_mean = torch.mean(targets, dim=1, keepdim=True)
    
    preds_centered = preds - preds_mean
    targets_centered = targets - targets_mean
    
    covariance = torch.sum(preds_centered * targets_centered, dim=1)
    preds_std = torch.sqrt(torch.sum(preds_centered ** 2, dim=1) + 1e-8)
    targets_std = torch.sqrt(torch.sum(targets_centered ** 2, dim=1) + 1e-8)
    
    correlation = covariance / (preds_std * targets_std)
    loss = -torch.mean(correlation)
    return loss

# 평균 상관계수 계산 함수 정의
def calculate_mean_correlation(predictions, targets):
    predictions = predictions.cpu().detach().numpy()
    targets = targets.cpu().detach().numpy()
    
    correlations = []
    for i in range(predictions.shape[1]):
        corr, _ = pearsonr(predictions[:, i], targets[:, i])
        correlations.append(corr)
    
    mean_correlation = np.mean(correlations)
    return mean_correlation

def covariance_loss(y_true, y_pred):
    # y_true와 y_pred의 형태: (batch_size, num_genes)
    batch_size = y_true.size(0)
    
    # 평균 제거
    y_true_centered = y_true - y_true.mean(dim=0, keepdim=True)
    y_pred_centered = y_pred - y_pred.mean(dim=0, keepdim=True)
    
    # 공분산 행렬 계산
    cov_true = (y_true_centered.t() @ y_true_centered) / (batch_size - 1)
    cov_pred = (y_pred_centered.t() @ y_pred_centered) / (batch_size - 1)
    
    # 프로베니우스 노름 계산
    cov_loss = torch.norm(cov_true - cov_pred, p='fro')
    
    return cov_loss

def cosine_similarity_loss(y_true, y_pred):
    # y_true와 y_pred의 형태: (batch_size, num_genes)
    y_true_normalized = nn.functional.normalize(y_true, p=2, dim=1)
    y_pred_normalized = nn.functional.normalize(y_pred, p=2, dim=1)
    
    # 코사인 유사도 계산
    cosine_similarity = (y_true_normalized * y_pred_normalized).sum(dim=1)
    loss = 1 - cosine_similarity  # 코사인 유사도를 최대화하기 위해
    return loss.mean()

def correlation_loss(y_true, y_pred):
    # y_true와 y_pred의 형태: (batch_size, num_genes)
    batch_size = y_true.size(0)
    
    # 평균 제거
    y_true_centered = y_true - y_true.mean(dim=0, keepdim=True)
    y_pred_centered = y_pred - y_pred.mean(dim=0, keepdim=True)
    
    # 표준편차 계산
    y_true_std = y_true_centered.std(dim=0, unbiased=False) + 1e-8
    y_pred_std = y_pred_centered.std(dim=0, unbiased=False) + 1e-8
    
    # 상관계수 행렬 계산
    corr_true = (y_true_centered.t() @ y_true_centered) / ((batch_size - 1) * y_true_std.unsqueeze(1) * y_true_std)
    corr_pred = (y_pred_centered.t() @ y_pred_centered) / ((batch_size - 1) * y_pred_std.unsqueeze(1) * y_pred_std)
    
    # 상관계수 차이의 제곱합
    corr_diff = corr_true - corr_pred
    loss = torch.sum(corr_diff ** 2)
    
    return loss
    
    
def get_model_path(model_dir, wandb_name, fold):
    pattern = f"{wandb_name}_fold{fold}"
    
    for file_name in os.listdir(model_dir):
        if pattern in file_name:
            return os.path.join(model_dir, file_name)
        
