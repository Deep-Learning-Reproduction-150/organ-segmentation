"""
This file contains a Dataset class that follows the PyTorch logic

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

import os
import importlib
import torch.nn as nn
from torch.utils.data import Dataset
from src.utils import bcolors, Logger
from src.Data.LabeledSample import LabeledSample
from src.Data.utils import DataTransformer


class CTDataset(Dataset):
    """
    The Data Loader can load data from folders and return a list of images
    represented by objects of the type CTData
    """

    # Attribute stores a global label structure to apply for every sample
    label_structure = None

    # Stores the location of the raw data set
    root = None

    # Attribute containing a list of provided samples
    samples = None

    # Stores the transforms to be applied to the data set
    label_transforms = None
    sample_transforms = None

    def __init__(self, root, preload: bool = True, label_transforms: list = None, output_transforms: list = None,
                 sample_transforms: list = None, no_logging: bool = True, label_structure: list = None):
        """
        Constructor method of the dataloader. First parameter specifies directories that contain labels to the files
        provided. The dataloader is designed to work with nrrd files only at the moment. n-dimensional numpy arrays
        could be included later on as well.

        :param root: where the data is stored (directory containing directories)
        :param label_folder_name: folder that contains labels
        :param preload: when true, system will load data directly when instantiating an object
        :param no_logging: in case you create a CTDataset in CLI or so, you don't want logging - turn off this way
        :param transforms: a transformer (can be composed from many before passing it to constructor)
        """

        # Call super class constructor
        super().__init__()

        # Save the transform TODO: remove it here later
        self.label_transforms = []
        self.sample_transforms = []
        self.output_transforms = []

        # Save the label structure
        self.label_structure = label_structure

        # Check if given path leads to a directory
        if not os.path.isdir(root):
            # Raise exception that the path is wrong
            raise ValueError(bcolors.FAIL + "ERROR: Given path is not a directory. Provide valid data directory." + bcolors.ENDC)

        # Remember the location of the data
        self.root = root

        # Get the count of files in the path (potential objects
        possible_target_count = len(os.listdir(self.root))

        # Print loading message
        if not no_logging:
            Logger.log("Started loading the data set with possibly " + str(possible_target_count) + " samples " +
                       "(preloading " + ("active" if preload else "inactive") + ")", type="INFO", in_cli=True)

        # Initiate the sample attribute
        self.samples = []

        # Create a counter variable
        counter = 0

        # Log the start of creating the instance
        if not no_logging:
            Logger.log("Attempt to generate a dataset instance", in_cli=True)

        # Save all the samples
        for i, element in enumerate(os.scandir(self.root)):

            # Check if the element is a directory (wanted structure for a labeled entry)
            if element.is_dir():

                # Create a new instance of a labeled sample
                new_sample = LabeledSample(
                    path=element.path,
                    label_transformer=self.get_data_transformer('labels'),
                    sample_transformer=self.get_data_transformer('sample')
                )

                # Append a labeled sample object
                self.samples.append(new_sample)

                # Increment the counter
                counter += 1

            # Print error of unexpected file in the passed directory
            if element.is_file() and not no_logging:
                # Log warning
                Logger.log("Unexpected file was found in data directory (" + str(element) + ")", type="WARNING", in_cli=True)

            if not no_logging:
                # Print the changing import status line
                done = (i / possible_target_count) * 100
                # Finish the status bar
                Logger.print_status_bar(done=done, title="creating dataset")

        # Show the 100% status bar
        if not no_logging:
            Logger.print_status_bar(done=100, title="creating dataset")
            Logger.end_status_bar()

        # If preloading is active, load the sample here already
        if preload:

            # Log the start of creating the instance
            if not no_logging:
                Logger.log("Start the preprocessing of the data", in_cli=True)
                Logger.print_status_bar(done=0, title="preprocessing")

            # Iterate through all samples
            for i, sample in enumerate(self.samples):

                # Load data for this sample and transform it
                sample.load()

                # Print the changing import status line
                if not no_logging:
                    done = ((i + 1) / len(self.samples)) * 100
                    Logger.print_status_bar(done=done, title="preprocessing")

            # End the status bar
            if not no_logging:
                Logger.end_status_bar()

        # Log that preloading was successful
        if not no_logging:
            Logger.log("Loading of data completed", type="SUCCESS", in_cli=True)

    def __getitem__(self, index):
        """
        This method returns a random example of the data set

        :param index: the index of the data sample
        :return: the labeled example at a specific index
        """

        # Get the sample with a certain index
        sample_instance = self.samples[index]

        # Create sample data (squeeze the dummy channel in there as well)
        sample = sample_instance.get_tensor()
        labels = sample_instance.get_labels(self.label_structure)

        # Check if there are any output transforms to apply
        if self.output_transforms is not None:
            output_transformer = self.get_data_transformer('output')
            output_transformer.output_mode = True
            sample = output_transformer(sample)
            labels = output_transformer(labels)

        # Return (data tensor, label tensor)
        return sample, labels

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

        TODO: also visualize the labels in here as well
        """

        # Iterate through all samples
        for index, sample in enumerate(self.samples):

            # Load the sample TODO: pass transformer rather to visualize functino (for inplace)
            sample.load(sample_transformer=self.get_data_transformer('sample'), label_transformer=self.get_data_transformer('labels'))

            # Create visualization for the sample
            sample.visualize(
                export_gif=True,
                direction=direction,
                high_quality=False,
                name="Sample " + str(sample.id),
                show_status_bar=True
            )

    def get_dataset_path(self):
        """
        Method returns the path where this data set is obtained from

        :return: path of the data set
        """

        # Return the root path
        return self.root

    def get_data_transformer(self, destination: str = 'sample'):
        """
        Returns an instance of a data transformer that contains the specified transformations
        """
        # Create transform set
        if destination == 'sample':
            transforms = self.sample_transforms
        elif destination == 'output':
            transforms = self.output_transforms
        else:
            transforms = self.label_transforms
        # Create a data transformer
        transform_list = []
        for t in transforms:
            if isinstance(t, dict):
                t = CTDataset.get_transform(**t)
            elif not isinstance(t, object) or not isinstance(t, nn.Module):
                raise TypeError('Expected type dict or transform.')
            transform_list.append(t)
        return DataTransformer(transform_list)

    @staticmethod
    def get_transform(name=None, **params):
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