import torch
import torch.nn as nn


class GiouLoss(nn.Module):
    def __init__(self, task='giou', reduction='none'):
        super(GiouLoss, self).__init__()
        self.reduction = reduction
        self.task = task

    def forward(self, inputs, targets):
        max_left = torch.max(inputs[:, 0], targets[:, 0])
        min_right = torch.min(inputs[:, 1], targets[:, 1])
        max_right = torch.max(inputs[:, 1], targets[:, 1])
        min_left = torch.min(inputs[:, 0], targets[:, 0])

        iou = (min_right - max_left) / (max_right - min_left + 1e-30)

        if self.task == 'iou':
            iou[iou < 0] = 0
        loss = 1 - iou
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
