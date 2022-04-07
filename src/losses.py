"""
This file contains different losses that can be set in a job

Course: Deep Learning
Date: 03.04.2022
Group: 150
"""
import copy

import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from matplotlib import pyplot as plt

# Default AC values
# background, brain stem, optic chiasma, mandible, optic nerve left, optic nerve right, parotid gland left, parotid gland right, submandibular left, submandibular right
# 0.5, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 1.0, 3.0, 3.0
# ORDERED!!!


DEFAULT_AC = torch.Tensor(
    [0.5, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 1.0, 3.0, 3.0]
)  # TODO: focal loss weights per channels from the paper


class CombinedLoss(nn.Module):

    input_dim = None
    alpha_vals = None

    def __init__(self, alpha, **kwargs):
        """
        TODO: Implement weights["focal"] as
        0.5, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 1.0, 3.0, and 3.0
        for
        background, brain stem, optic chiasma, mandible, optic nerve left, optic nerve right, parotid gland left, parotid gland right, submandibular left, submandibular right
        """
        super(CombinedLoss, self).__init__()

        self.dice = DiceLoss()
        self.focal = FocalLoss()
        self.alpha = alpha

    def forward(self, inputs, targets, l=1.0, gamma=2):

        # Get dice and focal loss
        dice = self.dice(inputs, targets)
        focal = self.focal(inputs, targets, alpha=self.get_alpha(inputs), gamma=gamma)

        # Stich them together and return
        combined = focal + l * dice
        return combined

    def get_alpha(self, inputs):
        if CombinedLoss.alpha_vals is None:
            alpha_tensor = torch.Tensor(self.alpha)
            placeholder = torch.ones_like(inputs)
            alpha = (placeholder.transpose(1, -1) * alpha_tensor).transpose(1, -1).view(-1)
            CombinedLoss.alpha_vals = alpha
        return CombinedLoss.alpha_vals


class DiceCoefficient(nn.Module):
    def __init__(self, **params):
        super().__init__()

    def forward(self, inputs, targets):
        # Compute the dice coefficient
        channels = inputs.size()[1]
        inputs = inputs[:].contiguous().view(-1)
        targets = targets[:].contiguous().view(-1)
        intersection = (inputs * targets).sum()
        dice = ((2.0 * intersection) / (inputs.sum() + targets.sum())) / channels
        return dice


class DiceLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()

    def forward(self, inputs, targets):
        dice = DiceCoefficient()(inputs, targets)
        return 1 - dice


class FocalLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()

    def forward(self, inputs, targets, alpha, gamma=2.0):

        # Remember the number of batches
        batches = inputs.size()[0]

        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, weight=alpha[: inputs.shape[0]], reduction="mean")

        BCE_EXP = torch.exp(-BCE)
        focal_loss = ((1 - BCE_EXP) ** gamma * BCE) / batches
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


class L1Loss(nn.Module):
    def __init__(self, **params):
        super().__init__()

    def forward(self, output_batch, input_batch):
        # Determine loss
        loss = nn.L1Loss()(output_batch, input_batch)
        return loss


class CrossEntropyLoss(nn.Module):
    def __init__(self, **params):
        super().__init__()

    def forward(self, output_batch, input_batch):
        # Determine loss
        loss = nn.CrossEntropyLoss()(output_batch, input_batch)
        return loss
