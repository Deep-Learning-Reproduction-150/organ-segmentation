"""
This file contains classes and functionality regarding data loading

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

import nrrd
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class ComputerTomographyImage:
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
            raise ValueError("Given path does not lead to a nrrd file")

        # Try to load the data at the given path
        try:

            # Load the data and throw it into an ndarray
            extracted_data, header = nrrd.read(path)

            # Save the as attributes for this instance
            self.data = extracted_data
            self.meta = header

            # Check if data dimensions are correct
            if self.meta['dimension'] != 3:
                raise ValueError("ERROR: file " + path + " contains " + str(self.meta['dimension']) + "-dimensional data (not expected 3D data)")

        except Exception as error:

            # Raise exception that file could not be loaded
            raise ValueError("ERROR: could not read nrrd file at " + path + "(" + str(error) + ")")

        # Save the path to the raw image data
        self.location = path

        # Extract the name of the file from the path where it is located
        filename = self.location.split('/')[-1]
        self.name = filename.split('.')[0]

    def get_data(self):
        """
        This method returns a three dimensional ndarray that contains the data

        :return: raw ndarray data
        """

        # Return the raw 3D nrarray
        return self.data

    def visualize(self):
        """
        Visualize the data using matplotlib

        TODO: Pretty sure, that this would not work
        """

        # Extract x y and z datapoints
        x, y, z = self.data.nonzero()

        # TODO: impossible to plot ~23mio datapoints like this ...
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(x, y, -z, zdir='z', c='red')
        # plt.savefig(self.name + ".png")


class LabeledSample:
    """
    This represents one labeled sample from the data. A sample contains raw data and the associated
    labels. In this context, those labels are itself 3D maps where specific organs are located at.
    """

    # This attribute stores the ComputerTomographyImage
    sample = None

    # This list stores the labels (also of type ComputerTomographyImage)
    labels = None

    # Attribute that stores the path to the folder that contains the sample data
    path = None

    def __init__(self, path, labels_folder_path: str = "structures"):
        """
        Constructor of the LabeledSample object. Expected by default is a folder that contains one nrrd file which
        is the sample data and a folder with name <labels_folder_path> that contains n labels, itself encoded as nrrd
        files.
        """

        # Check if this file exists
        if not os.path.isdir(path):
            # Raise an exception for this issue
            raise ValueError("ERROR: Given path does not lead to a folder")

        # Check if this file exists
        if not os.path.isdir(os.path.join(path, labels_folder_path)):
            # Raise an exception for this issue
            raise ValueError("ERROR: labels_folder_path is not valid (not found)")

        # Check if the folder is encoded in the expected format
        if len(glob.glob(path + '/*.nrrd')) > 1:
            # Print error that more then one data file was found
            raise Exception("ERROR: more than one sample data file found during creation of LabeledSample")
        else:
            # Create the sample CT file instance
            self.sample = ComputerTomographyImage(glob.glob(path + '/*.nrrd')[0])

        # Initiate a sample list
        self.labels = []

        # Iterate through the labels and create CT image instances for them as well
        for element in glob.glob(os.path.join(path, labels_folder_path) + '/*.nrrd'):
            # Create a label for storing
            label = ComputerTomographyImage(element)
            # Store the label in the labels attribute
            self.labels.append(label)


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
            raise ValueError("ERROR: Given path is not a directory. Provide valid data directory.")

        # Remember the location of the data
        self.path = path_to_samples

        # Initiate the sample attribute
        self.samples = []

        # Save all the samples
        for element in os.scandir(self.path):

            # Check if the element is a directory (wanted structure for a labeled entry)
            if element.is_dir():

                # Append a labeled sample object
                self.samples.append(LabeledSample(element.path))

            # Print error of unexpected file in the passed directory
            if element.is_file():

                # Display a warning about unexpected file in the specified data directory
                print("WARNING: Unexpected file was found in data directory (" + str(element) + ")")

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
