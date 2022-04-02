"""
This file contains the LabeledSample, an object that contains raw data as well as a set of labels. If it
is already labeled (training data), the labels are contained. They can however, also be added to this object
by the OrganNet25D.

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

import os
import glob
from torch import from_numpy
from src.utils import bcolors
from src.Dataloader.CTData import CTData


class LabeledSample:
    """
    This represents one labeled sample from the data. A sample contains raw data and the associated
    labels. In this context, those labels are itself 3D maps where specific organs are located at.
    """

    # A static attribute for auto increment primary key
    id_store = 1

    # Attribute storing the id of this sample
    id = None

    # This attribute stores the CTData
    sample = None

    # This list stores the labels (also of type CTData)
    labels = None

    # Attribute that stores the path to the folder that contains the sample data
    path = None

    # Whether or whether not to preload data
    preload = None

    def __init__(self, path, preload: bool = True, labels_folder_path: str = "structures"):
        """
        Constructor of the LabeledSample object. Expected by default is a folder that contains one nrrd file which
        is the sample data and a folder with name <labels_folder_path> that contains n labels, itself encoded as nrrd
        files.

        :param path: the path to the folder that contains the files
        :param preload: whether to load the data directly when instantiating an object
        :param labels_folder_path: folder within path that contains files with labels
        """

        # Save whether sample should reload data
        self.preload = preload

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
            self.sample = CTData(glob.glob(path + '/*.nrrd')[0], preload=self.preload)

        # Initiate a sample list
        self.labels = []

        # Iterate through the labels and create CT image instances for them as well
        for element in glob.glob(os.path.join(path, labels_folder_path) + '/*.nrrd'):
            # Create a label for storing
            label = CTData(element, preload=self.preload)
            # Store the label in the labels attribute
            self.labels.append(label)

    def visualize(self, show: bool = False, export_png: bool = False, export_gif: bool = False,
                  direction: str = "vertical", name: str = None, high_quality: bool = False,
                  show_status_bar: bool = True):
        """
        This method visualizes the labeled data sample. Vision is to have a great visualization of
        the raw data and the labeled regions (like brain stem and so on).

        :return: shows a visualization of the data
        """

        # Create name for this
        sample_name = name if name is not None else "sample_" + str(self.id)

        # Check given parameters
        self.sample.visualize(export_png=export_png, export_gif=export_gif, direction=direction, name=sample_name,
                              high_quality=high_quality, show=show, show_status_bar=show_status_bar)

    def get_tensor(self):
        """
        This method returns a tensor that contains the data of this sample

        :return tensor: which contains the data points
        """

        # Return the sample (which is a tensor)
        return self.sample.get_tensor()

    def get_labels(self):
        """
        This method returns the list of labels associated with this sample

        :return labels: list of tensors that are the labels

        TODO: checkout exactly how the logic shall work - where are the label names!?
        """

        # Initialize a list of labels
        label_tensors = []

        # Iterate through the labels and get tensors of each
        for label in self.labels:
            # Append a tensor
            label_tensors.append(label.get_tensor())

        # Return the list of labels
        return label_tensors

    def add_label(self, label: CTData):
        """
        This method can be used to add a computed label to this sample. That is done in inference mode by the
        OrganNet25D

        :param label: an instance of a computer tomography data
        """

        # Append the given label to this sample object
        self.labels.append(label)
