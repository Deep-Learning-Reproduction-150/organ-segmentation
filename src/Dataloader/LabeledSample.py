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
from src.utils import bcolors
from src.Dataloader import ComputerTomographyData


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
                  direction: str = "vertical", name: str = None, high_quality: bool = False):
        """
        This method visualizes the labeled data sample. Vision is to have a great visualization of
        the raw data and the labeled regions (like brain stem and so on).

        :return: shows a visualization of the data
        """

        # Create name for this
        sample_name = name if name is not None else "sample_" + str(self.id)

        # Check given parameters
        self.sample.visualize(export_png=export_png, export_gif=export_gif, direction=direction, name=sample_name,
                              high_quality=high_quality, show=show)

    def add_label(self, label: ComputerTomographyData):
        """
        This method can be used to add a computed label to this sample. That is done in inference mode by the
        OrganNet25D

        :param label: an instance of a computer tomography data
        """

        # Append the given label to this sample object
        self.labels.append(label)
