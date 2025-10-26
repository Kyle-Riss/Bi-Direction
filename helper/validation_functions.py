"""
검증 함수들
YoloLSTM 프로젝트의 validation_functions.py 구조를 참고하여 작성
"""
import torch
import torch.nn as nn
import numpy as np


def calculate_mae(pred, target):
    """Mean Absolute Error 계산"""
    return torch.mean(torch.abs(pred - target)).item()


def calculate_mse(pred, target):
    """Mean Squared Error 계산"""
    return torch.mean((pred - target) ** 2).item()


def calculate_rmse(pred, target):
    """Root Mean Squared Error 계산"""
    mse = calculate_mse(pred, target)
    return np.sqrt(mse)


def calculate_accuracy(pred, target, threshold=0.1):
    """정확도 계산 (threshold 내에 있는 예측의 비율)"""
    errors = torch.abs(pred - target)
    correct = torch.all(errors < threshold, dim=1)
    return torch.mean(correct.float()).item()


def get_classification_accuracy(pred, target):
    """분류 정확도 계산 (YoloLSTM 스타일)"""
    # 회귀 문제의 경우 MAE를 정확도로 사용
    return calculate_mae(pred, target)


def evaluate_model(model, dataloader, device, loss_function=None):
    """모델 평가 함수"""
    model.eval()
    total_loss = 0.0
    pred_list = []
    target_list = []
    
    with torch.no_grad():
        for data in dataloader:
            inputs = data[0].to(device)
            targets = data[1].to(device)
            
            if hasattr(model, 'forward') and 'targets' in model.forward.__code__.co_varnames:
                loss, outputs = model(inputs, targets=targets)
            else:
                outputs = model(inputs)
                if loss_function is not None:
                    loss = loss_function(outputs, targets)
                else:
                    loss = nn.MSELoss()(outputs, targets)
            
            total_loss += loss.item()
            pred_list.append(outputs)
            target_list.append(targets)
    
    if pred_list:
        pred_list = torch.cat(pred_list)
        target_list = torch.cat(target_list)
    
    avg_loss = total_loss / len(dataloader)
    mae = calculate_mae(pred_list, target_list)
    rmse = calculate_rmse(pred_list, target_list)
    accuracy = calculate_accuracy(pred_list, target_list)
    
    return {
        'loss': avg_loss,
        'mae': mae,
        'rmse': rmse,
        'accuracy': accuracy
    }
