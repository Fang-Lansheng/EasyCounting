import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Normalize


class UniformLoss(nn.Module):
    def __init__(self, ksize=16):
        super(UniformLoss, self).__init__()
        self.ksize = ksize

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.interpolate(x, (64, 64), mode='bilinear', align_corners=True)
        x = Normalize(0, 1)(x)

        kh, kw = self.ksize, self.ksize     # kernel size
        dh, dw = self.ksize, self.ksize     # stride

        patches = x.unfold(2, kh, dh).unfold(3, kw, dw)
        patches = patches.reshape(batch_size, -1, kh * kw)      # [B, N, L]
        patches = Normalize(0, 1)(patches)

        N, L = patches.shape[-2:]

        patches -= torch.mean(patches, dim=2, keepdim=True)     # [B, N, L]
        patches_t = torch.transpose(patches, 1, 2)              # [B, L, N]

        c = torch.matmul(patches, patches_t) / (L - 1)
        d = torch.stack([torch.diag(a) for a in c + 1e-8], dim=0)
        std_dev = torch.pow(d, 0.5)

        c = c / (std_dev.view(batch_size, -1, 1) + 1e-8)
        c = c / (std_dev.view(batch_size, 1, -1) + 1e-8)
        c = torch.clamp(c, -1.0, 1.0)

        return c.flatten(-1).mean(-1).mean()


# Based on Pearson product-moment correlation coefficient matrix
class CoefficientMatrixLoss(nn.Module):
    def __init__(self, ksize=16):
        super(CoefficientMatrixLoss, self).__init__()
        self.ksize = ksize

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.interpolate(x, (64, 64), mode='bilinear', align_corners=True)
#         x = Normalize(0, 1)(x)

        kh, kw = self.ksize, self.ksize     # kernel size
        dh, dw = self.ksize, self.ksize     # stride

        patches = x.unfold(2, kh, dh).unfold(3, kw, dw)
        patches = patches.reshape(batch_size, -1, kh * kw)      # [B, N, L]
        patches = Normalize(0, 1)(patches)

        N, L = patches.shape[-2:]

        patches -= torch.mean(patches, dim=2, keepdim=True)     # [B, N, L]
        patches_t = torch.transpose(patches, 1, 2)              # [B, L, N]

        c = torch.matmul(patches, patches_t) / (L - 1)
        d = torch.stack([torch.diag(a) for a in c + 1e-8], dim=0)
        std_dev = torch.pow(d, 0.5)

        c = c / (std_dev.view(batch_size, -1, 1) + 1e-8)
        c = c / (std_dev.view(batch_size, 1, -1) + 1e-8)
        c = torch.clamp(c, -1.0, 1.0)

        return c.flatten(-1).mean(-1).mean()


# Normalized Absolute Average Deviation
class NAADLoss(nn.Module):
    def __init__(self):
        super(NAADLoss, self).__init__()

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W

        x_flatten = torch.flatten(x, start_dim=1)               # [B, N]
        mean = torch.mean(x_flatten, dim=1, keepdim=True)       # [B, 1]
        loss = torch.sum(torch.abs(x_flatten - mean),
                         dim=1, keepdim=True)                   # [B, 1]
        loss /= (N * mean + 1e-8)
        loss = torch.sum(loss) / B

        return loss


# Standard Deviation
class STDLoss(nn.Module):
    def __init__(self):
        super(STDLoss, self).__init__()

    def forward(self, x):
        B, _, H, W = x.shape
        N = H * W

        x_flatten = torch.flatten(x, start_dim=1)               # [B, N]
        mean = torch.mean(x_flatten, dim=1, keepdim=True)       # [B, 1]

        loss = torch.pow(x_flatten - mean, 2)
        loss = torch.sqrt(torch.sum(loss) / (N * B + 1e-8))

        return loss


# Index of dispersion
class IDLoss(nn.Module):
    def __init__(self, in_size=(64, 64), grid_size=(4, 4)):
        super(IDLoss, self).__init__()
        self.in_size = in_size
        self.grid_size = grid_size

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.interpolate(x, self.in_size,
                          mode='bilinear', align_corners=True)

        q = self.grid_size[0] * self.grid_size[1]
        kh = int(self.in_size[0] / self.grid_size[0])
        kw = int(self.in_size[1] / self.grid_size[1])
        dh = int(self.in_size[0] / self.grid_size[0])
        dw = int(self.in_size[1] / self.grid_size[1])

        patches = x.unfold(2, kh, dh).unfold(3, kw, dw)
        patches = patches.reshape(batch_size, -1, kh * kw)

        patches_sum = patches.sum(2)
        patches_mean = patches_sum.mean(1)
        patches_std = patches_sum.std(1)

        loss = torch.pow(patches_std, 2) * (q - 1)
        loss = loss / (patches_mean + 1e-8)
        loss = torch.sum(loss) / batch_size

        return loss


# Global Shannon entropy
class GSELoss(nn.Module):
    def __init__(self, in_size=(64, 64), grid_size=(4, 4)):
        super(GSELoss, self).__init__()
        self.in_size = in_size
        self.grid_size = grid_size

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.interpolate(x, self.in_size,
                          mode='bilinear', align_corners=True)

        q = self.grid_size[0] * self.grid_size[1]
        kh = int(self.in_size[0] / self.grid_size[0])
        kw = int(self.in_size[1] / self.grid_size[1])
        dh = int(self.in_size[0] / self.grid_size[0])
        dw = int(self.in_size[1] / self.grid_size[1])

        patches = x.unfold(2, kh, dh).unfold(3, kw, dw)
        patches = patches.reshape(batch_size, -1, kh * kw)

        patches_sum = patches.sum(2) + 1e-8
        p = patches_sum / (patches_sum.sum(dim=1, keepdims=True))

        loss = torch.sum(p * torch.log10(p), dim=1, keepdim=True)
        loss = -1 * loss / np.log10(q)
        loss = 1 - torch.sum(loss) / batch_size

        return loss


def tensorNormalization(x):
    # type: (torch.Tensor) -> torch.Tensor
    B, _, H, W = x.shape

    x = x.view(B, -1)
    val_min = torch.min(x, dim=-1, keepdim=True)[0]
    val_max = torch.max(x, dim=-1, keepdim=True)[0]
    x = torch.div(torch.sub(x, val_min),
                  torch.sub(val_max, val_min))
    x = x.view(B, -1, H, W)

    return x


def cov(m, rowvar=False):
    ''' Estimate a covariance matrix given data. '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()
