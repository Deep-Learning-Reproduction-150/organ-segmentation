"""
This file contains different losses that can be set in a job

Course: Deep Learning
Date: 03.04.2022
Group: 150
"""
import copy
from multiprocessing.sharedctypes import Value

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import CrossEntropyLoss
from torch.nn import L1Loss

from matplotlib import pyplot as plt

# Default AC values
# background, brain stem, optic chiasma, mandible, optic nerve left, optic nerve right, parotid gland left, parotid gland right, submandibular left, submandibular right
# 0.5, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 1.0, 3.0, 3.0
# ORDERED!!!


DEFAULT_AC = torch.Tensor(
    [0.5, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 1.0, 3.0, 3.0]
)  # TODO: focal loss weights per channels from the paper


class CrossEntropyLoss(CrossEntropyLoss):
    def __init__(self, *args, **kwargs):
        super().__init__()


class L1Loss(L1Loss):
    def __init__(self, *args, **kwargs):
        super().__init__()


class CombinedLoss(nn.Module):
    def __init__(self, alpha, **kwargs):
        """
        TODO: Implement weights["focal"] as
        0.5, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 1.0, 3.0, and 3.0
        for
        background, brain stem, optic chiasma, mandible, optic nerve left, optic nerve right, parotid gland left, parotid gland right, submandibular left, submandibular right
        """
        super(CombinedLoss, self).__init__()
        self.input_dim = None
        self.alpha_vals = None
        self.dice = DiceCoefficient()
        self.focal = FocalLoss()
        self.alpha = alpha

    def forward(self, inputs, targets, l=1.0, gamma=2, dsc_reduce="mean", return_per_channel_dsc=False):

        # Get dice and focal loss
        dice, dice_per_channel = self.dice(
            inputs, targets, reduce_method=dsc_reduce, return_per_channel_dsc=return_per_channel_dsc
        )
        focal = self.focal(inputs, targets, alpha=self.get_alpha(inputs), gamma=gamma)

        # Stich them together and return
        loss = focal + l * (1 - dice)

        if return_per_channel_dsc:
            return loss, dice_per_channel
        return loss

    def get_alpha(self, inputs):
        if self.alpha_vals is None:
            alpha_tensor = torch.Tensor(self.alpha)
            placeholder = torch.ones_like(inputs)
            alpha = (placeholder.transpose(1, -1) * alpha_tensor).transpose(1, -1).view(-1)
            self.alpha_vals = alpha
        return self.alpha_vals


class DiceCoefficient(nn.Module):
    def __init__(self, **params):
        super().__init__()

    def forward(self, inputs, targets, reduce_method="mean", return_per_channel_dsc=False):
        # Compute the dice coefficient
        # channels = inputs.size()[1]
        # inputs = inputs[:].contiguous().view(-1)
        # targets = targets[:].contiguous().view(-1)
        # intersection = (inputs * targets).sum()
        # dice = ((2.0 * intersection) / (inputs.sum() + targets.sum())) / channels
        # return dice
        # Compute the elementwise operations p * y and p + y
        dice_top = 2 * inputs * targets + 1e-4
        dice_bottom = inputs + targets + 1e-4
        dice = dice_top / dice_bottom
        if reduce_method == "mean":
            dsc_per_channel = dice.mean(dim=(0, 3, 2, 4))
        elif reduce_method == "sum":
            dsc_per_channel = dice.sum(dim=(0, 3, 2, 4))
        else:
            raise ValueError("Unrecognized reduce_method")

        dsc_avg = dsc_per_channel.mean()

        if return_per_channel_dsc:
            return dsc_avg, dsc_per_channel

        return dsc_avg


class DiceLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()

    def forward(self, inputs, targets, dsc_reduce_method="mean", return_per_channel_dsc=False):
        dice = DiceCoefficient()(
            inputs, targets, reduce_method=dsc_reduce_method, return_per_channel_dsc=return_per_channel_dsc
        )
        if return_per_channel_dsc:
            loss, per_channel = dice
            return 1 - loss, per_channel

        return 1 - loss


class FocalLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()

    def forward(self, inputs, targets, alpha, gamma=2.0):

        # # Remember the number of batches
        # batches = inputs.size()[0]

        # # flatten label and prediction tensors
        # inputs = inputs.view(-1)
        # targets = targets.view(-1)

        # # first compute binary cross-entropy
        # BCE = F.binary_cross_entropy(inputs, targets, weight=alpha[: inputs.shape[0]], reduction="mean")

        # BCE_EXP = torch.exp(-BCE)
        # focal_loss = ((1 - BCE_EXP) ** gamma * BCE) / batches
        # return focal_loss
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        weight = alpha[: inputs.shape[0]] * (1 - targets) ** gamma
        focal_loss = F.binary_cross_entropy(inputs, targets, weight=weight, reduction="mean")
        return focal_loss


class MSELoss(nn.Module):
    """
    Determines l1 loss which is the absolute difference between input and output.
    """

    def __init__(self, **params):
        """
        Initialize method of the MSE Loss object

        :param reduction:   'mean' will determine the mean loss over all elements (across batches) while
                            'sum' will determine the summation of losses over all elements
        """
        super().__init__()
        self.reduction = "mean"

    def forward(self, output_batch, input_batch):

        # Determine loss
        loss = nn.MSELoss(reduction=self.reduction)(output_batch, input_batch)

        # In case of summation we want the batch loss, hence we divide by the batch size
        if self.reduction == "sum":
            loss = loss / input_batch.shape[0]

        return loss
