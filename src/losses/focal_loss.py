import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, n_classes, alpha=None, gamma=2, reduction='none'):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.ones(n_classes, 1)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        N, C = inputs.size(0), inputs.size(1)
        P = F.softmax(inputs, dim=-1)

        class_mask = inputs.data.new(N, C).fill_(0).to(device=inputs.device)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        self.alpha = self.alpha.to(device=inputs.device)
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)
        log_p = probs.log()
        loss = -alpha * (torch.pow((1-probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
