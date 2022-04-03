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

        # Obtain batch dimension
        batch_dimension = len(batch)

        # Channel dimension
        channel_dimension = 10

        # Obtain the desired 3D CT image dimensions
        x_dim = self.batch_dimensions[0]
        y_dim = self.batch_dimensions[1]
        z_dim = self.batch_dimensions[2]

        # TODO: generate a batch and label tensor based on loader

        # Create a random batch
        batch = torch.randn(batch_dimension, x_dim, y_dim, z_dim)

        # Generate a tensor that contains the labels
        labels = torch.randn(batch_dimension, x_dim, y_dim, z_dim, channel_dimension)

        return batch, labels