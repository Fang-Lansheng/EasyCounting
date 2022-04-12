import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize


class DiffLoss(nn.Module):
    def __init__(self, in_size=(64, 64), grid_size=(4, 4)):
        super(DiffLoss, self).__init__()
        self.in_size = in_size
        self.grid_size = grid_size

    def forward(self, x, y):
        batch_size = x.shape[0]
        x = F.interpolate(x, self.in_size, mode='bilinear', align_corners=True)
        y = F.interpolate(y, self.in_size, mode='bilinear', align_corners=True)

        q = self.grid_size[0] * self.grid_size[1]
        kh = int(self.in_size[0] / self.grid_size[0])
        kw = int(self.in_size[1] / self.grid_size[1])
        dh = int(self.in_size[0] / self.grid_size[0])
        dw = int(self.in_size[1] / self.grid_size[1])

        x_patches = x.unfold(2, kh, dh).unfold(3, kw, dw)
        x_patches = x_patches.reshape(batch_size, -1, kh * kw)
        y_patches = y.unfold(2, kh, dh).unfold(3, kw, dw)
        y_patches = y_patches.reshape(batch_size, -1, kh * kw)

        x_patches_sum = x_patches.sum(2) + 1e-16
        y_patches_sum = y_patches.sum(2) + 1e-16

        p = x_patches_sum / x_patches_sum.sum(dim=1, keepdims=True)

        loss = (p - 1/q) * (x_patches_sum - y_patches_sum) * q
        loss = torch.exp(-1 * loss.sum(1))
        loss = torch.sum(loss) / batch_size

        return loss
