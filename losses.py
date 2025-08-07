# losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, gamma=1.0, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.smooth= smooth

    def forward(self, preds, targets):
        # preds: [B,C,H,W] raw logits
        probs = F.softmax(preds, dim=1)
        C = probs.shape[1]
        # one-hot targets [B,C,H,W]
        t = F.one_hot(targets, C).permute(0,3,1,2).float()
        # compute per-class TP, FP, FN
        TP = (probs * t).sum(dim=(2,3))
        FP = (probs * (1-t)).sum(dim=(2,3))
        FN = ((1-probs) * t).sum(dim=(2,3))
        Tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)
        return torch.mean((1 - Tversky) ** self.gamma)

# 如果想同时加上 CrossEntropy，可这么写：
class CombinedLoss(nn.Module):
    def __init__(self, w_ce=1.0, w_tv=1.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.tv = FocalTverskyLoss()
        self.w_ce, self.w_tv = w_ce, w_tv
    def forward(self, preds, targets):
        return self.w_ce*self.ce(preds, targets) + self.w_tv*self.tv(preds, targets)
