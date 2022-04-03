"""
This file contains different losses that can be set in a job

Course: Deep Learning
Date: 03.04.2022
Group: 150
"""

import torch.nn as nn


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
        self.reduction = 'mean'

    def forward(self, output_batch, input_batch):

        # Determine loss
        loss = nn.MSELoss(reduction=self.reduction)(output_batch, input_batch)

        # In case of summation we want the batch loss, hence we divide by the batch size
        if self.reduction == 'sum':
            loss = loss / input_batch.shape[0]

        return loss


class DiceLoss(nn.Module):
    """
    Computes the dice loss
    """

    def __init__(self, **params):
        """
        Initializes the dice score

        TODO: params contains everything that is passed via job setup
        """
        super().__init__()

    def forward(selfself, output_batch, input_batch):

        a = 0
        # TODO: write dice loss for this

        return output_batch
