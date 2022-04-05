import torch
from torch import nn
from torchvision.transforms import CenterCrop


"""
Contains auxiliary functions for the model.
"""


def crop3d(x: torch.Tensor, target_shape: tuple):
    """
    Assuming (..., D, H, H) input and shapes.
    """
    out_xycropped = CenterCrop(target_shape[-1])(x)
    diff = (x.shape[-3] - target_shape[-3]) // 2
    if diff != 0:
        out_xyzcropped = out_xycropped[:, :, diff:-diff, ...]
    else:
        out_xyzcropped = out_xycropped
    return out_xyzcropped


def conv_2x2d(
    in_channels=1, out_channels=16, groups=1, kernel_size=(1, 3, 3), stride=1, padding="valid", *args, **kwargs
):
    return nn.Sequential(
        nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            *args,
            **kwargs,
        ),
        nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            groups=groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            *args,
            **kwargs,
        ),
        nn.ReLU(),  # Maybe ReLU, maybe something else
    )


def conv_2x3d_coarse(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0, *args, **kwargs):
    """
    The 2xConv with 3,3,3 kernel without the ResSE presented in the paper
    """
    return nn.Sequential(
        nn.intrinsic.ConvBnReLU3d(  # does this speed things up?
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            torch.nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ),
        nn.intrinsic.ConvBnReLU3d(  # does this speed things up?
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            torch.nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ),
    )


def HDC(in_channels=64, out_channels=128, dilations=(1, 2, 5), kernel_size=(3, 3, 3)):
    """
    Creates a HDC layer.
    """
    layers = []
    prev_out = in_channels
    for dilation in dilations:
        layer = nn.Conv3d(
            in_channels=prev_out,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding="same",
            dilation=dilation,
        )
        layers.append(layer)
        prev_out = out_channels
    hdc = nn.Sequential(*layers)
    return hdc
