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
from src.utils import bcolors, print_status_bar
from src.Dataloader.LabeledSample import LabeledSample


class CTDataset(Dataset):
    """
    The Data Loader can load data from folders and return a list of images
    represented by objects of the type CTData

    TODO:
        - How can this be used smartly in training
        - Are lists efficient? Should there be transformation already?

    """

    # Stores the location of the raw data set
    path = None

    # Attribute representing usage of cross validation
    cross_validate = None

    # Attribute containing a list of provided samples
    samples = None

    def __init__(self, path_to_samples, label_folder_name: str = "structures", use_cross_validation: bool = True):
        """
        Constructor method of the dataloader. First parameter specifies directories that contain labels to the files
        provided. The dataloader is designed to work with nrrd files only at the moment. n-dimensional numpy arrays
        could be included later on as well.

        :param label_folder_name: folder that contains labels
        :param path_to_samples: where the data is stored (directory containing directories)

        TODO: think about passing the transformers here to stick to PyTorch logic
        """

        # Check if given path leads to a directory
        if not os.path.isdir(path_to_samples):
            # Raise exception that the path is wrong
            raise ValueError(bcolors.FAIL + "ERROR: Given path is not a directory. Provide valid data directory." + bcolors.ENDC)

        # Remember the location of the data
        self.path = path_to_samples

        # Get the count of files in the path (potential objects
        possible_target_count = len(os.listdir(self.path))

        # Print message
        print(bcolors.OKCYAN + "INFO: Started loading the data set with possibly " + str(possible_target_count) +
              " samples ..." + bcolors.ENDC)

        # Initiate the sample attribute
        self.samples = []

        # Create a counter variable
        counter = 0

        # Save all the samples
        for i, element in enumerate(os.scandir(self.path)):

            # Check if the element is a directory (wanted structure for a labeled entry)
            if element.is_dir():

                # Append a labeled sample object
                self.samples.append(LabeledSample(path=element.path, labels_folder_path=label_folder_name))

                # Increment the counter
                counter += 1

            # Print error of unexpected file in the passed directory
            if element.is_file():

                # Display a warning about unexpected file in the specified data directory
                warning = bcolors.WARNING + "WARNING: Unexpected file was found in data directory (" + str(element) + ")" + bcolors.ENDC
                sys.stdout.write("\r" + warning)
                sys.stdout.flush()
                print("")

            # Print the changing import status line
            done = (i / possible_target_count) * 100
            # Finish the status bar
            print_status_bar(done=done, title="imported")

        # Reset console for next print message
        print_status_bar(done=100, title="imported")

        # Save whether the dataset should utilize cross validation
        self.cross_validate = use_cross_validation

        # Display details regarding data loading
        print("INFO: Done loading the dataset at " + self.path + " (found and loaded " + str(counter) + " samples)")

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
