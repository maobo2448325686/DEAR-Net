import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-3):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predict, target):
        intersection = torch.sum(predict * target)
        union = torch.sum(predict) + torch.sum(target) + self.smooth
        dice = (2. * intersection + self.smooth) / union
        return 1 - dice
