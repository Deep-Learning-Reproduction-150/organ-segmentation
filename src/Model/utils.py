import torch
from torch import nn
from torchvision.transforms import CenterCrop

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
