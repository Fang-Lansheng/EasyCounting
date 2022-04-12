import torch
import torch.nn as nn
from torch.nn import init
from einops import rearrange


class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, activation=None, use_bn=False):
        super(Conv2D, self).__init__()

        self.activation = activation
        self.use_bn = use_bn

        kernel_size = (kernel_size, kernel_size)
        dilation = (dilation, dilation)
        stride = (1, 1)
        padding = tuple((k - 1) * d // 2 for (k, d) in zip(kernel_size, dilation))

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation)

        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_channels)

        self._init_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class Conv3D(nn.Module):
    def __init__(self, in_channels, out_channels,
                 kernel_size=(3, 3, 3), dilation=(1, 1, 1),
                 activation=None, use_bn=False):
        super(Conv3D, self).__init__()

        stride = (1, 1, 1)
        padding = tuple((k - 1) * d // 2 for (k, d) in zip(kernel_size, dilation))

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation)

        self.activation = activation
        self.use_bn = use_bn

        if self.use_bn:
            self.bn = nn.BatchNorm3d(out_channels)

        self._init_weights()

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.activation:
            x = self.activation(x)

        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class SelfAttention(nn.Module):
    def __init__(self, d_model, dim_heads=64, heads=8, dropout=0.1):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.scale = dim_heads ** -0.5

        self.fc_q = nn.Linear(d_model, dim_heads * heads)
        self.fc_k = nn.Linear(d_model, dim_heads * heads)
        self.fc_v = nn.Linear(d_model, dim_heads * heads)
        self.fc_o = nn.Linear(dim_heads * heads, d_model)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def forward(self, x):       # [B, L, C]
        q = self.fc_q(x)        # [B, L, D]
        k = self.fc_k(x)        # [B, L, D]
        v = self.fc_v(x)        # [B, L, D]

        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)  # [B, h, L, d]
        k = rearrange(k, 'b n (h d) -> b h d n', h=self.heads)  # [B, h, d, L]
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)  # [B, h, L, d]

        att = torch.matmul(q, k) * self.scale                   # [B, h, L, L]
        att = torch.softmax(att, -1)
        att = self.dropout(att)

        out = torch.matmul(att, v)                              # [B, h, L, d]
        out = rearrange(out, 'b h n d -> b n (h d)')            # [B, L, D]
        out = self.fc_o(out)                                    # [B, L, C]

        return out

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # init.kaiming_normal_(m.weight, mode='fan_out')
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
