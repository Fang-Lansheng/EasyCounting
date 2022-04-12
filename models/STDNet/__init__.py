import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from einops import rearrange
from misc.nn import Conv2D, Conv3D


class STDNet(nn.Module):
    def __init__(self, num_blocks=4, use_bn=False):
        super(STDNet, self).__init__()

        if use_bn:
            self.Backbone = models.vgg16_bn(pretrained=True).features[:33]
        else:
            self.Backbone = models.vgg16(pretrained=True).features[:23]

        self.DenseBlocks = nn.ModuleList([DSTB(use_bn=use_bn)] * num_blocks)

        self.Head = nn.Sequential(
            Conv2D(512, 128, 3, 1, activation=nn.ReLU(), use_bn=use_bn),
            Conv2D(128, 64, 3, 1, activation=nn.ReLU(), use_bn=use_bn),
            Conv2D(64, 1, 1))

    def forward(self, x):
        # Resize & Adjust channels
        B, T, C, H, W = x.shape                 # [B, T, C, H, W]
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = F.interpolate(x, (360, 640), mode='bilinear', align_corners=False)

        # Backbone
        x = self.Backbone(x)                    # [B * T, 512, H/2, W/2]

        # Dense Blocks
        x = rearrange(x, '(b t) c h w -> b t c h w', t=T)
        for block in self.DenseBlocks:
            x = block(x)                        # [B, T, 512, h, w]

        # Regression Head
        x = rearrange(x, 'b t c h w -> (b t) c h w')
        x = self.Head(x)                        # [B * T, 1, h, w]

        x = F.interpolate(x, (H, W), mode='bilinear', align_corners=False)
        x = x.reshape(B, T, -1, H, W)           # [B, T, 1, H, W]

        return x


class DSTB(nn.Module):
    def __init__(self, use_bn=False):
        super(DSTB, self).__init__()

        self.DSB = DenseSpatialBlock(use_bn=use_bn)
        self.DTB = DenseTemporalBlock(use_bn=use_bn)

    def forward(self, x):
        B, T, _, H, W = x.shape         # [B, T, 512, H, W]

        x = x.reshape(B * T, -1, H, W)  # [B * T, 512, H, W]
        x = self.DSB(x)                 # [B * T, 512, H, W]

        x = x.reshape(B, T, -1, H, W)   # [B, T, 512, H, W]
        x = x.permute(0, 2, 1, 3, 4)    # [B, 512, T, H, W]
        x = self.DTB(x.contiguous())    # [B, 512, T, H, W]

        x = x.permute(0, 2, 1, 3, 4)    # shape same as input

        return x.contiguous()           # [B, T, 512, H, W]


class DenseSpatialBlock(nn.Module):
    def __init__(self, use_bn=True):
        super(DenseSpatialBlock, self).__init__()

        self.conv1_1 = Conv2D(512, 256, 1, 1, activation=nn.ReLU(), use_bn=use_bn)
        self.conv1_2 = Conv2D(256,  64, 3, 1, activation=nn.ReLU(), use_bn=use_bn)

        self.conv2_1 = Conv2D(576, 256, 1, 1, activation=nn.ReLU(), use_bn=use_bn)
        self.conv2_2 = Conv2D(256,  64, 3, 2, activation=nn.ReLU(), use_bn=use_bn)

        self.conv3_1 = Conv2D(640, 256, 1, 1, activation=nn.ReLU(), use_bn=use_bn)
        self.conv3_2 = Conv2D(256,  64, 3, 3, activation=nn.ReLU(), use_bn=use_bn)

        self.conv4_1 = Conv2D(704, 512, 1, 1, activation=nn.ReLU(), use_bn=use_bn)
        self.att = SpatialChannelAwareBlock()

    def forward(self, x):                       # [N, 512, H, W]    N = B * T
        z1 = self.conv1_1(x)                    # [N, 256, H, W]
        z1 = self.conv1_2(z1)                   # [N,  64, H, W]

        z2 = torch.cat([x, z1], dim=1)          # [N, 576, H, W]
        z2 = self.conv2_1(z2)                   # [N, 256, H, W]
        z2 = self.conv2_2(z2)                   # [N,  64, H, W]

        z3 = torch.cat([x, z1, z2], dim=1)      # [N, 640, H, W]
        z3 = self.conv3_1(z3)                   # [N, 256, H, W]
        z3 = self.conv3_2(z3)                   # [N,  64, H, W]

        z4 = torch.cat([x, z1, z2, z3], dim=1)  # [N, 704, H, W]
        z4 = self.conv4_1(z4)                   # [N, 512, H, W]

        z4 = self.att(z4)                       # [N, 512, H, W]  same as input

        return z4


class DenseTemporalBlock(nn.Module):
    def __init__(self, use_bn=True):
        super(DenseTemporalBlock, self).__init__()

        self.conv1_1 = Conv3D(512, 256, (1, 1, 1), (1, 1, 1), activation=nn.ReLU(), use_bn=use_bn)
        self.conv1_2 = Conv3D(256,  64, (3, 1, 1), (1, 1, 1), activation=nn.ReLU(), use_bn=use_bn)

        self.conv2_1 = Conv3D(576, 256, (1, 1, 1), (1, 1, 1), activation=nn.ReLU(), use_bn=use_bn)
        self.conv2_2 = Conv3D(256,  64, (3, 1, 1), (2, 1, 1), activation=nn.ReLU(), use_bn=use_bn)

        self.conv3_1 = Conv3D(640, 256, (1, 1, 1), (1, 1, 1), activation=nn.ReLU(), use_bn=use_bn)
        self.conv3_2 = Conv3D(256,  64, (3, 1, 1), (3, 1, 1), activation=nn.ReLU(), use_bn=use_bn)

        self.conv4_1 = Conv3D(704, 512, (1, 1, 1), (1, 1, 1), activation=nn.ReLU(), use_bn=use_bn)
        self.att = TemporalChannelAwareBlock()

    def forward(self, x):                       # [B, 512, T, H, W]
        z1 = self.conv1_1(x)                    # [B, 256, T, H, W]
        z1 = self.conv1_2(z1)                   # [B,  64, T, H, W]

        z2 = torch.cat([x, z1], dim=1)          # [B, 576, T, H, W]
        z2 = self.conv2_1(z2)                   # [B, 256, T, H, W]
        z2 = self.conv2_2(z2)                   # [B,  64, T, H, W]

        z3 = torch.cat([x, z1, z2], dim=1)      # [B, 640, T, H, W]
        z3 = self.conv3_1(z3)                   # [B, 256, T, H, W]
        z3 = self.conv3_2(z3)                   # [B,  64, T, H, W]

        z4 = torch.cat([x, z1, z2, z3], dim=1)  # [B, 704, T, H, W]
        z4 = self.conv4_1(z4)                   # [B, 512, T, H, W]

        z4 = self.att(z4)                       # [B, 512, T, H, W]  same as input

        return z4


class SpatialChannelAwareBlock(nn.Module):
    def __init__(self):
        super(SpatialChannelAwareBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.FC = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.Sigmoid())

        self._init_weights()

    def forward(self, x):
        N, C, H, W = x.shape

        a = self.avg_pool(x).view(N, C)
        a = self.FC(a).view(N, C, 1, 1)

        x = x * a.expand_as(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class TemporalChannelAwareBlock(nn.Module):
    def __init__(self):
        super(TemporalChannelAwareBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        self.FC = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
            nn.Sigmoid())

        self._init_weights()

    def forward(self, x):
        B, C, T, H, W = x.shape

        a = self.avg_pool(x).view(B, C)
        a = self.FC(a).view(B, C, 1, 1, 1)

        x = x * a.expand_as(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
