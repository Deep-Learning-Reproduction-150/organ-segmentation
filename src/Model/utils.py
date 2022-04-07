import torch
from torch import nn
from torchvision.transforms import CenterCrop

from HDC import ConvBNReLU

"""
Contains auxiliary functions for the model.
"""


def activation_mapper(s: str) -> nn.Module:
    mapper = {
        None: None,
        "none": None,
        "relu": nn.ReLU(),
        "ReLU": nn.ReLU(),
        "sigmoid": nn.Sigmoid(),
        "tanh": nn.Tanh(),
        "softmax": nn.Softmax(dim=1),
    }
    return mapper[s]


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
    in_channels=1,
    out_channels=16,
    groups=1,
    kernel_size=(1, 3, 3),
    stride=1,
    padding="valid",
    activation=nn.ReLU(),
    *args,
    **kwargs
):
    return nn.Sequential(
        ConvBNReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            padding=padding,
        ),
        ConvBNReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            padding=padding,
        ),
    )


def conv_2x3d_coarse(
    in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0, activation="linear", *args, **kwargs
):
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
