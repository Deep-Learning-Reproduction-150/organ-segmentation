"""
This file contains helpers for the data loading processes

Course: Deep Learning
Date: 03.04.2022
Group: 150
"""

import torch


class CustomCompose(object):
    """
    Custom compose transform to make use of mask.
    """

    def __init__(self, transforms, return_mask=False):
        self.transforms = transforms
        self.return_mask = return_mask

    def __call__(self, img, mask=None):
        if mask is None:
            for t in self.transforms:
                img = t(img)
            return img
        for t in self.transforms:
            img, mask = t(img, mask)
        if self.return_mask:
            return img, mask
        return img


class CTDataCollator(object):
    """
    This custom collate function is used by the data loader to create batches
    """

    # Storing the desired batch dimensions
    batch_dimensions = None

    def __init__(self, batch_dimensions: tuple):
        """
        Constructor method

        :param batch_dimensions: desired batch dimensions
        """

        # Save the desired dimensions
        self.batch_dimensions = batch_dimensions

    def __call__(self, batch):
        """
        Method returns batch tensor from data set subset

        :param batch: subset of the data set, based on sampler
        :return: tensor tupel which represents a batch and the label tensor
        """

        return batch
