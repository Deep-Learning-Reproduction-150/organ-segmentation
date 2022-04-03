"""
This file contains a Dataset class that follows the PyTorch logic

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

import os
import random
import sys
from torch.utils.data import DataLoader, Dataset
from src.utils import bcolors, Logger
from src.Dataloader.LabeledSample import LabeledSample


class CTDataset(Dataset):
    """
    The Data Loader can load data from folders and return a list of images
    represented by objects of the type CTData

    TODO:
        - How can this be used smartly in training
        - Implement preloading or on demand loading (if set)
        - Are lists efficient? Should there be transformation already?

    """

    # Stores the location of the raw data set
    root = None

    # Attribute containing a list of provided samples
    samples = None

    # Whether to preload the data or load it when obtaining a sample
    preload = None

    # Stores the transform object
    transform = None

    # Whether the data set has already been transformed
    transformed = None

    def __init__(self, root, label_folder_name: str = "structures", preload: bool = True, transform: list = []):
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

        # Save the transform TODO: handle the transforms
        self.transform = transform

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
            Logger.print_status_bar(done=100, title="imported")
            Logger.end_status_bar()

        # Only show status bar when preloading
        if self.preload:

            # Display details regarding data loading
            Logger.log("Done loading the dataset at " + self.root + " (found and loaded " + str(counter) + " samples)", in_cli=True)

    def __getitem__(self, index):
        """
        This method returns a random example of the data set

        :param index: the index of the data sample
        :return: a random example from the data set

        TODO:
            - also do the transformations, maybe initially passed to the dataset?
            - what about the labels? how do you return multi-labels?
        """

        # TODO: Call transform (to assure that data has been transformed)

        # Get the sample with a certain index
        sample = self.samples[index]

        # Return the tupel (data, labels)
        return sample.get_tensor(), sample.get_labels()

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

    def check_dimensionality(self, remove_false_samples: bool = False):
        """
        Method scans all samples and checks if the dimensions add up

        :param remove_false_samples: removes samples when their dimensions dont add up

        TODO: maybe later use transform to unify them all?
        """

        # Initiate a sample shape
        global_shape = None

        shapes = []

        # Check for different dimensions
        for sample in self.samples:

            # Preprocess the sample
            sample.preprocess()
            # Get the dimensions
            sample_shape = sample.get_tensor().data.shape

            shapes.append(sample_shape)

        a = 0

        # Check for different dimensions
        for sample in self.samples:
            # Preprocess the sample
            sample.preprocess()
            # Get the dimensions
            sample_shape = sample.get_tensor().data.shape

            # Check if global shape is still none
            if global_shape is None:
                global_shape = sample_shape

            # Check if they differ
            if global_shape != sample_shape:

                # Check if shall remove false sample
                if remove_false_samples:
                    Logger.log("Sample " + sample.path + " has varying dimensions, it will be removed", type="WARNING", in_cli=True)
                    self.samples.remove(sample)
                else:
                    Logger.log("Sample " + sample.path + " has varying dimensions from the rest of the samples", type="ERROR", in_cli=True)
