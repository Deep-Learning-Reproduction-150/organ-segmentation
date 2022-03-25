"""
This file contains classes and functionality regarding data loading

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

import pandas as pd
import numpy as np


class ComputerTomographyImage:
    """
    This class represents a CT Image and is used to depict HaN CT images

    TODO:
        - What functionality should a CT Image have?

    """

    # This attribute stores the location of the images raw data
    location = None

    def __init__(self, path):
        """
        Constructor of a CT Image

        :param path:
        """

        # Save the path to the raw image data
        self.location = path


class DataLoader:
    """
    The Data Loader can load data from folders and return a list of images
    represented by objects of the type ComputerTomographyImage

    TODO:
        - How can this be used smartly in training
        - Are lists efficient? Should there be transformation already?

    """

    # Stores the location of the raw data set
    path = None

    # Attribute representing usage of cross validation
    cross_validate = None

    def __init__(self, path, use_cross_validation: bool = True):
        """
        Constructor method of the dataloader

        :param path: where the data is stored
        """

        # Remember the location of the data
        self.path = path

        # Save whether the dataset should utilize cross validation
        self.cross_validate = use_cross_validation

    def get_random_example(self, not_from_training_set: bool = False):
        """
        This method returns a random example of the data set

        :param not_from_training_set: if example shall not be from training set
        :return: a random example from the data set

        TODO:
            - Should there be any options here?
        """

        # TODO: implement cool picking of example

        return None

    def get_training_data(self):
        """
        Returns a set of training data

        :return: training data set
        """

        # Initialize a list of images
        images = []

        # TODO: load the data located at path
        # FIXME: what kind of format should be returned?

        return images

    def get_testing_data(self):
        """
        Returns a set of testing data

        :return: testing data set
        """

        # Initialize a list of images
        images = []

        # TODO: load the data located at path
        # FIXME: what kind of format should be returned?

        return images
