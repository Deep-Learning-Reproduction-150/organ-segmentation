"""
This file contains the Dataloader that reads data from folders and creates a usable structure from it

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

from torch.utils.data import DataLoader


class Dataloader(DataLoader):
    """
    The Data Loader can load data from folders and return a list of images
    represented by objects of the type CTData

    TODO:
        - Learn how PyTorch does it directly
    """

    a = 0