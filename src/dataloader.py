"""
This file contains classes and functionality regarding data loading

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

import os
import glob
import random
from src.helpers import bcolors, print_status_bar
import numpy as np
import sys
import pandas as pd
import nrrd
import imageio
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from IPython.display import Image as show_gif


class ComputerTomographyData:
    """
    This class represents a CT Image and is used to depict HaN CT images

    TODO:
        - What "else" functionality should a CT Image have?
    """

    # This attribute stores the location of the images raw data
    location = None

    # This attribute stores the data in a ndarray format
    data = None

    # The name of the file (used for labels as well)
    name = None

    # This meta data contains information about the data obtained from the input file
    meta = None

    def __init__(self, path):
        """
        Constructor of a CT Image

        :param path:
        """

        # Check if this file exists
        if not os.path.exists(path):
            # Raise an exception for this issue
            raise ValueError(bcolors.FAIL + "ERROR: Given path does not lead to a nrrd file" + bcolors.ENDC)

        # Try to load the data at the given path
        try:

            # Load the data and throw it into an ndarray
            extracted_data, header = nrrd.read(path)

            # Save the as attributes for this instance
            self.data = extracted_data
            self.meta = header

            # Check if the data has three dimensions
            if self.data.ndim != 3:
                raise ValueError(bcolors.FAIL + "ERROR: Unexpected number of dimensions (" + str(self.data.ndim) + ") in data sample" + bcolors.ENDC)

            # Check if data dimensions are correct
            if self.meta['dimension'] != 3:
                raise ValueError(bcolors.FAIL + "ERROR: file " + path + " contains " + str(self.meta['dimension']) + "-dimensional data (not expected 3D data)" + bcolors.ENDC)

        except Exception as error:

            # Raise exception that file could not be loaded
            raise ValueError(bcolors.FAIL + "ERROR: could not read nrrd file at " + path + "(" + str(error) + ")" + bcolors.ENDC)

        # Save the path to the raw image data
        self.location = path

        # Extract the name of the file from the path where it is located
        filename = self.location.split('/')[-1]
        self.name = filename.split('.')[0]

    def get_data(self):
        """
        This method returns a three dimensional ndarray that contains the data

        :return data: raw ndarray data
        """

        # Return the raw 3D ndarray
        return self.data

    def visualize(self, show: bool = False, export_png: bool = False, export_gif: bool = False,
                  direction: str = "vertical", name: str = None):
        """

        Visualize the data using matplotlib

        TODO: Pretty sure, that this would not work

        :param show: directly displays the images here
        :param name: either None (default name) or special name for file
        :param direction: how to go through image, options: vertical, horizontal
        :param export_png: whether system shall create png images for the slices
        :param export_gif: whether system shall create a GIF file from the data
        """

        # Check given parameters
        if direction not in ['vertical', 'horizontal']:
            raise ValueError(bcolors.FAIL + "ERROR: Direction has to either be 'vertical' or 'horizontal'" + bcolors.ENDC)

        # Print a status update
        print("INFO: Creating visualization of " + str(self.data.ndim) + "-dimensional data " + self.name + " with direction " + direction)

        # Extract the three dimensions from the data set
        shape = self.data.shape
        x_dimensions = shape[0]
        z_dimensions = shape[2]
        dim_counter = z_dimensions if direction == 'vertical' else x_dimensions

        # Filenames
        images = []

        # Iterate through all layers of the image
        for index in range(dim_counter):

            # Get the data from this layer
            slice_data = self.data[:, :, index] if direction == 'vertical' else self.data[index, :, :]

            # Create an image
            plt.figure(figsize=(14, 14))
            plt.gray()
            plt.imshow(slice_data)
            plt.draw()

            # Print additional status updates
            plt.title(name + ' (layer ' + str(index) + ')')
            plt.xlabel("x coordinate" if direction == "vertical" else "depth")
            plt.ylabel("y coordinate")

            if show:
                plt.show()

            # If export png is on, save export
            if export_png:

                # Check if the folder exists
                if not os.path.isdir('visualizations'):
                    # Create folder as it does not exist yet
                    os.mkdir('visualizations')

                # Folder name for the png output
                folder_name = 'visualizations/' + (self.name if name is None else name)

                # Check if the folder exists
                if not os.path.isdir(folder_name):
                    # Create folder as it does not exist yet
                    os.mkdir(folder_name)

                # Create a file for that image
                plt.savefig(folder_name + '/slice_' + str(index) + '.png', dpi=100)

            # Append this
            if export_gif:
                tmp_image_path = 'tmp.png'
                plt.savefig(tmp_image_path, dpi=100)
                images.append(imageio.imread(tmp_image_path))

            # Close the image
            plt.close()

            # Print the changing import status line
            done = (index / dim_counter) * 100
            print_status_bar(done=done, title="written")

        # Finish the status bar
        print_status_bar(done=100, title="written")

        # If system shall export a GIF from it, do so
        if export_gif:

            # Print status update
            print("INFO: Creating visualization of " + str(self.data.ndim) + "-dimensional data " + str(self.name) + ", saving GIF file")

            # Remove the tmp tile
            os.remove('tmp.png')

            # Check if the folder exists
            if not os.path.isdir('visualizations'):
                # Create folder as it does not exist yet
                os.mkdir('visualizations')

            # Save GIF file
            imageio.mimsave('visualizations/' + (self.name if name is None else name) + '.gif', images)


class LabeledSample:
    """
    This represents one labeled sample from the data. A sample contains raw data and the associated
    labels. In this context, those labels are itself 3D maps where specific organs are located at.
    """

    # A static attribute for auto increment primary key
    id_store = 1

    # Attribute storing the id of this sample
    id = None

    # This attribute stores the ComputerTomographyData
    sample = None

    # This list stores the labels (also of type ComputerTomographyData)
    labels = None

    # Attribute that stores the path to the folder that contains the sample data
    path = None

    def __init__(self, path, labels_folder_path: str = "structures"):
        """
        Constructor of the LabeledSample object. Expected by default is a folder that contains one nrrd file which
        is the sample data and a folder with name <labels_folder_path> that contains n labels, itself encoded as nrrd
        files.
        """

        # Assign an id and increment the id store
        self.id = LabeledSample.id_store
        LabeledSample.id_store += 1

        # Check if this file exists
        if not os.path.isdir(path):
            # Raise an exception for this issue
            raise ValueError(bcolors.FAIL + "ERROR: Given path does not lead to a folder" + bcolors.ENDC)

        # Check if this file exists
        if not os.path.isdir(os.path.join(path, labels_folder_path)):
            # Raise an exception for this issue
            raise ValueError(bcolors.FAIL + "ERROR: labels_folder_path is not valid (not found)" + bcolors.ENDC)

        # Check if the folder is encoded in the expected format
        if len(glob.glob(path + '/*.nrrd')) > 1:
            # Print error that more then one data file was found
            raise Exception(bcolors.FAIL + "ERROR: more than one sample data file found during creation of LabeledSample" + bcolors.ENDC)
        else:
            # Create the sample CT file instance
            self.sample = ComputerTomographyData(glob.glob(path + '/*.nrrd')[0])

        # Initiate a sample list
        self.labels = []

        # Iterate through the labels and create CT image instances for them as well
        for element in glob.glob(os.path.join(path, labels_folder_path) + '/*.nrrd'):
            # Create a label for storing
            label = ComputerTomographyData(element)
            # Store the label in the labels attribute
            self.labels.append(label)

    def visualize(self, show: bool = False, export_png: bool = False, export_gif: bool = False,
                  direction: str = "vertical", name: str = None):
        """
        This method visualizes the labeled data sample. Vision is to have a great visualization of
        the raw data and the labeled regions (like brain stem and so on).

        :return: shows a visualization of the data
        """

        # Create name for this
        sample_name = name if name is not None else "sample_" + str(self.id)

        # Check given parameters
        if direction not in ['vertical', 'horizontal']:
            raise ValueError(
                bcolors.FAIL + "ERROR: Direction has to either be 'vertical' or 'horizontal'" + bcolors.ENDC)

        # Make sure labels and image have the same shape
        for label in self.labels:
            if label.data.shape != self.sample.data.shape:
                raise ValueError(bcolors.FAIL + "ERROR: Label of " + sample_name + " does not have the same data dimensions as sample " + bcolors.ENDC)

        # Print a status update
        print("INFO: Creating visualization of " + sample_name + " with " + str(len(self.labels)) + " labels")

        # Extract the three dimensions from the data set
        shape = self.sample.data.shape
        x_dimensions = shape[0]
        z_dimensions = shape[2]
        dim_counter = z_dimensions if direction == 'vertical' else x_dimensions

        # Filenames
        images = []

        # Iterate through all layers of the image
        for index in range(dim_counter):

            # Get the data from this layer
            slice_data = self.sample.data[:, :, index] if direction == 'vertical' else self.sample.data[index, :, :]

            # Create an image
            plt.figure(figsize=(14, 14))
            plt.gray()
            plt.imshow(slice_data)

            # TODO: think of a method to join label and sample data in one figure
            # for label in self.labels:
            #     label_slice = label.data[:, :, index] if direction == 'vertical' else label.data[index, :, :]
            #     plt.imshow(label_slice)

            plt.draw()

            # Print additional status updates
            plt.title(name + ' (layer ' + str(index) + ')')
            plt.xlabel("x coordinate" if direction == "vertical" else "depth")
            plt.ylabel("y coordinate")

            if show:
                plt.show()

            # If export png is on, save export
            if export_png:

                # Check if the folder exists
                if not os.path.isdir('visualizations'):
                    # Create folder as it does not exist yet
                    os.mkdir('visualizations')

                # Folder name for the png output
                folder_name = 'visualizations/' + sample_name

                # Check if the folder exists
                if not os.path.isdir(folder_name):
                    # Create folder as it does not exist yet
                    os.mkdir(folder_name)

                # Create a file for that image
                plt.savefig(folder_name + '/slice_' + str(index) + '.png', dpi=100)

            # Append this
            if export_gif:
                tmp_image_path = 'tmp.png'
                plt.savefig(tmp_image_path, dpi=100)
                images.append(imageio.imread(tmp_image_path))

            # Close the image
            plt.close()

            # Print the changing import status line
            done = (index / dim_counter) * 100
            print_status_bar(done=done, title="written")

        # Finish the status bar
        print_status_bar(done=100, title="written")

        # If system shall export a GIF from it, do so
        if export_gif:

            # Print status update
            print("INFO: Creating visualization of " + sample_name + " with " + str(len(self.labels)) + " labels - composing a GIF")

            # Remove the tmp tile
            os.remove('tmp.png')

            # Check if the folder exists
            if not os.path.isdir('visualizations'):
                # Create folder as it does not exist yet
                os.mkdir('visualizations')

            # Save GIF file
            imageio.mimsave('visualizations/' + sample_name + '.gif', images)


class DataLoader:
    """
    The Data Loader can load data from folders and return a list of images
    represented by objects of the type ComputerTomographyData

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
                self.samples.append(LabeledSample(element.path))

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
        print("INFO: Done loading the dataset at " + self.path + " (contained " + str(counter) + " samples)")

    def get_random_example(self, not_from_training_set: bool = False):
        """
        This method returns a random example of the data set

        :param not_from_training_set: if example shall not be from training set
        :return: a random example from the data set
        """

        # Get a random sample from the dataset
        sample = random.sample(self.samples, 1)[0]

        # Return the sample
        return sample

    def create_all_visualizations(self, direction: str = "vertical"):
        """
        This method creates all visualizations for every sample in the dataset

        :param direction: either vertical or horizontal
        :return: writes a GIF for every sample
        """

        # Iterate through all samples
        for sample in self.samples:
            sample.visualize(export_gif=True, direction=direction, name="sample_" + str(sample.id))

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
