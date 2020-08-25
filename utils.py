import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn


'''
Numpy version for final evaluation
'''
def cv_rmse(y_pred, y_true):
    if type(y_pred) is np.ndarray:
        y_mean = np.mean(y_true)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    else:
        y_mean = torch.mean(y_true)
        rmse = torch.sqrt(nn.MSELoss()(y_pred, y_true))
    return rmse / y_mean

'''
Pytorch version for training
'''
def cv_rmse_loss(output, target, alpha=0.7, beta=0.3):
    if len(output.shape) == 3:
        output = output.resize(output.size(0) * output.size(1), output.size(2))
        target = target.resize(target.size(0) * target.size(1), target.size(2))
    w_mean = torch.mean(target[:, 0])
    q_mean = torch.mean(target[:, 1])
    w_rmse = torch.sqrt(nn.MSELoss()(output[:, 0], target[:, 0]))
    q_rmse = torch.sqrt(nn.MSELoss()(output[:, 1], target[:, 1]))
    loss = alpha*w_rmse/w_mean + beta*q_rmse/q_mean
    return loss

def train_loss(output, target, avg):
    y_mean = torch.mean(target)
    rmse_loss = torch.sqrt(nn.MSELoss()(output, target))
    avg_reg = torch.sqrt(nn.MSELoss()(torch.mean(output, dim=1), avg.squeeze(-1)))
    # ratio_reg = nn.MSELoss()(torch.max(output, dim=1)[0] / (torch.min(output, dim=1)[0] + 10), torch.tensor(5).float())
    # reg_loss = nn.MSELoss()(output[1:], output[:-1])
    loss = rmse_loss / y_mean + 0.01*avg_reg
    return loss