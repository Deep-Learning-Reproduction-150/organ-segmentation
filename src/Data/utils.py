"""
This file contains helpers for the data loading processes

Course: Deep Learning
Date: 03.04.2022
Group: 150
"""

import torch


class CTDataCollator(object):
    """
    This custom collate function is used by the data loader to create batches

    :param dataset: batch data ste
    :return: tensor for network
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
        :return: tensor which represents batch
        """

        # Create a random batch
        batch = torch.randn(128, 128, 128)

        # TODO: create a batch like really <3

        return batch
