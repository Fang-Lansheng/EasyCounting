import torch
import torch.nn as nn

from pytorch_msssim import ssim
from torchvision.transforms import Normalize


class SSIMLoss(nn.Module):
    def __init__(self, data_range=1.0, nonnegative_ssim=True):
        super(SSIMLoss, self).__init__()
        self.data_range = data_range
        self.nonnegative_ssim = nonnegative_ssim

    def forward(self, x, y):
        batch_size = x.shape[0]
        x = Normalize(0, 1)(x)
        y = Normalize(0, 1)(y)

        ssim_idx = ssim(x, y, data_range=self.data_range,
                        nonnegative_ssim=self.nonnegative_ssim)

        return (1 - ssim_idx) / batch_size


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
