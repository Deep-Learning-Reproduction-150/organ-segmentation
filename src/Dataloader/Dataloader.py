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

    def __init__(self, dataset, batch_size=4, shuffle=True, num_workers=0):
        super().__init__(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

        # TODO: do awesome stuff


# train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
# test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)