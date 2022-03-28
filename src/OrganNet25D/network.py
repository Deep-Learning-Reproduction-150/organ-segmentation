"""
This file contains the OrganNet25D that can be trained by using the dataloader and used
by just passing instances of ComputerTomographyData.

Course: Deep Learning
Date: 25.03.2022
Group: 150

TODO:
    - Should we create a class Organ? So that could be outputted really nicely?
"""

import torch
from src.Dataloader.ComputerTomographyData import ComputerTomographyData


class OrganNet25D:
    """
    This represents the OrganNet25D model as proposed by Chen et. al.

    TODO:
        - Basically implement everything :D
        - Would be really cool if this class already capsules some functionality
    """

    def __init__(self):
        """
        Constructor method of the OrganNet
        """

        in_channels = 25
        out_channels = 2

        # Add a layer for first block, 2D convolution
        self.block_one = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=12,
                kernel_size=(3, 3),
                padding=0,
                stride=(1, 1)
            ),
            torch.nn.MaxPool3d(
                kernel_size=(3, 3),
                padding=0,
                stride=(1, 1)
            ),
            # TODO: add more elements of block 1
        )

        # Add a layer for second block, Coarse 3D convolution
        self.block_two = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=12,
                kernel_size=(3, 3, 3),
                padding=0,
                stride=(1, 1)
            ),
            # TODO: add more elements of block 2
        )

        # Add a layer for third block, Fine 3D convolution
        self.block_three = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3, 3),
                padding=0,
                stride=(1, 1)
            ),
            # TODO: add more elements of block 3
        )

    def train(self, train_data, test_data, monitor_progress: bool = False):
        """
        This method trains the network automatically

        :param monitor_progress: if user wants detailed feedback on learning progress
        :param train_data: a data set that contains examples for training
        :param test_data: a data set that contains examples for testing
        :return: if validation was successful

        TODO:
            - How to proceed with e.g. cross validation
            - Is the paper already forcing a specific mode of operation?
            - Write validation of the trained model
        """

        # Check if system shall monitor learning progress
        if monitor_progress:

            # TODO: write some really informative progress overview
            a = 0

        # TODO: write training algorithm

        return True

    def forward(self, x: ComputerTomographyData):
        """
        This method takes an image of type ComputerTomographyData and
        uses the model to create the segmentation of organs

        :param x: an instance of ComputerTomographyData
        :return: TODO: good question - what exactly?
        """

        # forward sample through network
        x = self.block_one.forward(x)
        x = self.block_two.forward(x)
        x = self.block_three.forward(x)

        # TODO: what is this outcome going to be? How can this be most useful?

        # Return the computed outcome
        return x

    def get_organ_segments(self, x: ComputerTomographyData):
        """
        This method returns the actual organ segments and not raw data of the inputted
        ComputerTomographyData

        :param x: an example of type ComputerTomographyData
        :return: the organs detected in the image
        """

        # TODO: use forward method and stuff to output really nice segment representation

        return None
