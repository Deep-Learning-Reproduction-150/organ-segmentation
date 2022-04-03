"""
This file contains a Dataset class that follows the PyTorch logic

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

import os
import random
import sys
import importlib
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from src.utils import bcolors, Logger
from src.Data.LabeledSample import LabeledSample
from src.Data.utils import DataTransformer


class CTDataset(Dataset):
    """
    The Data Loader can load data from folders and return a list of images
    represented by objects of the type CTData

    TODO:
        - How can this be used smartly in training
        - Implement preloading or on demand loading (if set)
        - Are lists efficient? Should there be transformation already?

    """

    # Attribute stores a global label structure to apply for every sample
    label_structure = []

    # Stores the location of the raw data set
    root = None

    # Attribute containing a list of provided samples
    samples = None

    # Whether to preload the data or load it when obtaining a sample
    preload = None

    # Stores the transforms to be applied to the data set
    transforms = None

    # Whether the data set has already been transformed
    transformed = None

    def __init__(self, root, label_folder_name: str = "structures", preload: bool = True, transforms: list = []):
        """
        Constructor method of the dataloader. First parameter specifies directories that contain labels to the files
        provided. The dataloader is designed to work with nrrd files only at the moment. n-dimensional numpy arrays
        could be included later on as well.

        :param root: where the data is stored (directory containing directories)
        :param label_folder_name: folder that contains labels
        :param preload: when true, system will load data directly when instantiating an object
        :param transform: a transformer (can be composed from many before passing it to constructor)

        TODO: think about passing the transformers here to stick to PyTorch logic
        """

        # Call super class constructor
        super().__init__()

        # Initiate transformed with false
        self.transformed = False

        # Whether or not to preload and preprocess volumes
        self.preload = preload

        # Save the transform
        self.transforms = transforms

        # Check if given path leads to a directory
        if not os.path.isdir(root):
            # Raise exception that the path is wrong
            raise ValueError(bcolors.FAIL + "ERROR: Given path is not a directory. Provide valid data directory." + bcolors.ENDC)

        # Remember the location of the data
        self.root = root

        # Get the count of files in the path (potential objects
        possible_target_count = len(os.listdir(self.root))

        # Only show status bar when preloading
        if self.preload:

            # Print loading message
            Logger.log("Started loading the data set with possibly " + str(possible_target_count) +
                       " samples ...", type="INFO", in_cli=True)

        # Initiate the sample attribute
        self.samples = []

        # Create a counter variable
        counter = 0

        # Save all the samples
        for i, element in enumerate(os.scandir(self.root)):

            # Check if the element is a directory (wanted structure for a labeled entry)
            if element.is_dir():

                # Append a labeled sample object
                self.samples.append(LabeledSample(path=element.path, preload=self.preload, labels_folder_path=label_folder_name))

                # Increment the counter
                counter += 1

            # Print error of unexpected file in the passed directory
            if element.is_file():
                # Log warning
                Logger.log("Unexpected file was found in data directory (" + str(element) + ")", type="WARNING", in_cli=True)

            # Only show status bar when preloading
            if self.preload:

                # Print the changing import status line
                done = (i / possible_target_count) * 100
                # Finish the status bar
                Logger.print_status_bar(done=done, title="imported")

        # Reset console for next print message
        if self.preload:
            # Show the 100% status bar
            Logger.print_status_bar(done=100, title="imported")
            Logger.end_status_bar()

        # Obtain one unified label structure
        for s in self.samples:
            for l in s.labels:
                if l.name not in CTDataset.label_structure:
                    CTDataset.label_structure.append(l.name)

        # Only show status bar when preloading
        if self.preload:

            # Display details regarding data loading
            Logger.log("Done loading the dataset at " + self.root + " (" + str(counter) + " samples)", in_cli=True)

            # Already preprocess the data here
            for i, sample in enumerate(self.samples):
                Logger.print_status_bar(done=((i + 1) / len(self.samples))*100, title="transforming")
                sample.preprocess(self.get_data_transformer(), CTDataset.label_structure, output_info=False)
            Logger.end_status_bar()

    def __getitem__(self, index):
        """
        This method returns a random example of the data set

        :param index: the index of the data sample
        :return: a random example from the data set

        TODO:
            - also do the transformations, maybe initially passed to the dataset?
            - what about the labels? how do you return multi-labels?
        """

        # Get the sample with a certain index
        sample = self.samples[index]

        # Preprocess the data (if that has not happened before)
        sample.preprocess(self.get_data_transformer(), CTDataset.label_structure, output_info=True)

        # Create sample data (squeeze the dummy channel in there as well)
        sample_data = sample.transformed_sample.unsqueeze(0)

        # Return the tupel (data, labels)
        return sample_data, sample.transformed_labels

    def __len__(self):
        """
        This method returns the length of the data set

        :return length: of the dataset
        """

        # Return count of samples
        return len(self.samples)

    def create_all_visualizations(self, direction: str = "vertical"):
        """
        This method creates all visualizations for every sample in the dataset

        :param direction: either vertical or horizontal
        :return: writes a GIF for every sample in the data set
        """

        # Iterate through all samples
        for index, sample in enumerate(self.samples):

            # Create visualization
            sample.visualize(export_gif=True, direction=direction, high_quality=False,
                             name="Sample " + str(sample.id), show_status_bar=True)

    def get_dataset_path(self):
        """
        Method returns the path where this data set is obtained from

        :return: path of the data set
        """

        # Return the root path
        return self.root

    def get_transform(self, name=None, **params):
        """
        Returns a transform based on identifier. This method will first look for a
        local transform in utils.transforms and secondly, look for an official
        pytorch transform.
        """
        # Try to import local custom module
        try:
            module = importlib.import_module('src.Data.transforms')
            transform = getattr(module, name)
        # Try to import pytorch transform
        except AttributeError:
            module = importlib.import_module('torchvision.transforms')
            transform = getattr(module, name)
        return transform(**params)

    def get_data_transformer(self):
        """
        Returns a list or single transform object based on a list or single transform description as dict.
        """
        # Create a data transformer
        transform_list = []
        for t in self.transforms:
            if isinstance(t, dict):
                t = self.get_transform(**t)
            elif not isinstance(t, object) or not isinstance(t, nn.Module):
                raise TypeError('Expected type dict or transform.')
            transform_list.append(t)
        return DataTransformer(transform_list)
