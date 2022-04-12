"""
Distribution Matching for Crowd Counting

Paper:
    B. Wang, H. Liu, D. Samaras, and M. H. Nguyen, “Distribution Matching for
    Crowd Counting,” in Advances in Neural Information Processing Systems,
    2020, vol. 33.

GitHub repository:
    https://github.com/cvlab-stonybrook/DM-Count
"""

import torch.nn as nn
import torch.nn.functional as F
import torchvision


class DM_Count(nn.Module):
    def __init__(self):
        super(DM_Count, self).__init__()
        self.backbones = torchvision.models.vgg19(pretrained=True).features[:36]
        self.reg_layers = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x):
        x = self.backbones(x)
        x = F.upsample_bilinear(x, scale_factor=2)
        x = self.reg_layers(x)
        mu = self.density_layer(x)
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-8)
        return mu, mu_normed
