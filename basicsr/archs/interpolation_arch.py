"""Traditional interpolation methods as network architectures for baseline comparison.

These architectures contain no trainable parameters and serve as baselines
to evaluate the effectiveness of deep learning super-resolution methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class NearestInterpolation(nn.Module):
    """Nearest neighbor interpolation baseline.

    Args:
        upscale (int): Upsampling factor (2, 4, or 8).
        in_chans (int): Number of input channels (default: 2 for flow field data).
    """

    def __init__(self, upscale=2, in_chans=2, **kwargs):
        super(NearestInterpolation, self).__init__()
        self.upscale = upscale
        self.in_chans = in_chans

    def forward(self, x):
        """
        Args:
            x (Tensor): Input LQ image, shape (B, C, H, W).

        Returns:
            Tensor: Upsampled HR image, shape (B, C, H*upscale, W*upscale).
        """
        return F.interpolate(
            x,
            scale_factor=self.upscale,
            mode='nearest'
        )


@ARCH_REGISTRY.register()
class BilinearInterpolation(nn.Module):
    """Bilinear interpolation baseline.

    Args:
        upscale (int): Upsampling factor (2, 4, or 8).
        in_chans (int): Number of input channels (default: 2 for flow field data).
    """

    def __init__(self, upscale=2, in_chans=2, **kwargs):
        super(BilinearInterpolation, self).__init__()
        self.upscale = upscale
        self.in_chans = in_chans

    def forward(self, x):
        """
        Args:
            x (Tensor): Input LQ image, shape (B, C, H, W).

        Returns:
            Tensor: Upsampled HR image, shape (B, C, H*upscale, W*upscale).
        """
        return F.interpolate(
            x,
            scale_factor=self.upscale,
            mode='bilinear',
            align_corners=False
        )


@ARCH_REGISTRY.register()
class BicubicInterpolation(nn.Module):
    """Bicubic interpolation baseline.

    This is the most commonly used traditional baseline in super-resolution literature.

    Args:
        upscale (int): Upsampling factor (2, 4, or 8).
        in_chans (int): Number of input channels (default: 2 for flow field data).
    """

    def __init__(self, upscale=2, in_chans=2, **kwargs):
        super(BicubicInterpolation, self).__init__()
        self.upscale = upscale
        self.in_chans = in_chans

    def forward(self, x):
        """
        Args:
            x (Tensor): Input LQ image, shape (B, C, H, W).

        Returns:
            Tensor: Upsampled HR image, shape (B, C, H*upscale, W*upscale).
        """
        return F.interpolate(
            x,
            scale_factor=self.upscale,
            mode='bicubic',
            align_corners=False
        )
