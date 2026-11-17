import numpy as np
import torch

def smape_np(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    diff = np.abs(y_true - y_pred)
    return float(np.mean(np.where(denom == 0, 0.0, diff / denom)) * 100.0)

@torch.no_grad()
def smape_torch(y_true, y_pred, eps: float = 1e-8):
    denom = (torch.abs(y_true) + torch.abs(y_pred)).clamp_min(eps) / 2.0
    diff = torch.abs(y_pred - y_true)
    return (diff / denom).mean() * 100.0
