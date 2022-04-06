"""
This file contains the LabeledSample, an object that contains raw data as well as a set of labels. If it
is already labeled (training data), the labels are contained. They can however, also be added to this object
by the OrganNet25D.

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

from copy import deepcopy
import os
import glob
import torch
from torch import from_numpy
from src.utils import bcolors, Logger
from src.Data.CTData import CTData
from src.Data.utils import DataTransformer


class LabeledSample:
    """
    This represents one labeled sample from the data. A sample contains raw data and the associated
    labels. In this context, those labels are itself 3D maps where specific organs are located at.
    """

    # A static attribute for auto increment primary key
    id_store = 1

    # Attribute storing the id of this sample
    id = None

    # This attribute stores the actual CTData (raw)
    sample = None

    # This list stores the labels (also of type CTData)
    labels = None

    # Attribute that stores the path to the folder that contains the sample data
    path = None

    # Stores whether the sample has been processed already
    loaded = None

    def __init__(self, path, labels_folder_path: str = "structures"):
        """
        Constructor of the LabeledSample object. Expected by default is a folder that contains one nrrd file which
        is the sample data and a folder with name <labels_folder_path> that contains n labels, itself encoded as nrrd
        files.

        :param path: the path to the folder that contains the files
        :param labels_folder_path: folder within path that contains files with labels
        """

        # Assign an id and increment the id store
        self.id = LabeledSample.id_store
        LabeledSample.id_store += 1

        # Initiate loaded with false
        self.loaded = False

        # Save the path to this sample
        self.path = path

        # Check if this file exists
        if not os.path.isdir(path):
            # Raise an exception for this issue
            raise ValueError(bcolors.FAIL + "ERROR: Given path does not lead to a folder" + bcolors.ENDC)

        # Check if this file exists
        if not os.path.isdir(os.path.join(path, labels_folder_path)):
            # Raise an exception for this issue
            raise ValueError(bcolors.FAIL + "ERROR: labels_folder_path is not valid (not found)" + bcolors.ENDC)

        # Check if the folder is encoded in the expected format
        if len(glob.glob(path + "/*.nrrd")) > 1:
            # Print error that more then one data file was found
            raise Exception(
                bcolors.FAIL
                + "ERROR: more than one sample data file found during creation of LabeledSample"
                + bcolors.ENDC
            )
        else:
            # Create the sample CT file instance
            self.sample = CTData(path=glob.glob(path + "/*.nrrd")[0])

        # Initiate a sample list
        self.labels = []

        # Iterate through the labels and create CT image instances for them as well
        for element in glob.glob(os.path.join(path, labels_folder_path) + "/*.nrrd"):
            # Create a label for storing
            label = CTData(path=element)
            # Store the label in the labels attribute
            self.labels.append(label)

    def visualize(
        self,
        show: bool = False,
        export_png: bool = False,
        export_gif: bool = False,
        direction: str = "vertical",
        name: str = None,
        high_quality: bool = False,
        show_status_bar: bool = True,
    ):
        """
        This method visualizes the labeled data sample. Vision is to have a great visualization of
        the raw data and the labeled regions (like brain stem and so on).

        :return: shows a visualization of the data
        """

        # Create name for this
        sample_name = name if name is not None else "sample_" + str(self.id)

        # Check given parameters
        self.sample.visualize(
            export_png=export_png,
            export_gif=export_gif,
            direction=direction,
            name=sample_name,
            high_quality=high_quality,
            show=show,
            show_status_bar=show_status_bar,
        )

    def get_tensor(self, transformer: DataTransformer = DataTransformer([])):
        """
        This method returns a tensor that contains the data of this sample

        :return tensor: which contains the data points
        """

        # Return the sample (which is a tensor)
        return self.sample.get_tensor(transformer=transformer)

    def get_labels(self, label_order: list, transformer: DataTransformer = DataTransformer([])):
        """
        This method returns the list of labels associated with this sample

        :return labels: list of tensors that are the labels
        """

        # Initialize a list of labels
        tensors = []

        # Iterate through the labels and create
        for wanted_label in label_order:

            # Iterate through the labels and find it
            data = None
            for label in self.labels:
                if label.name == wanted_label:
                    data = label.get_tensor(transformer=transformer, dtype=torch.int8)
                    break

            # Check if label exists
            if data is None:
                # Create zero sample
                if len(tensors) > 0:
                    # Use an existing label tensor as reference
                    data = torch.zeros_like(tensors[0], dtype=torch.int8)
                else:
                    # Have to obtain sample tensor (much more inefficient)
                    data = torch.zeros_like(self.get_tensor(transformer=transformer), dtype=torch.int8)

            # Append the transformed label to it
            tensors.append(data)

        # Compute a "background label" and append it to the labels
        label_mask = torch.ones_like(tensors[0], dtype=torch.int8)
        for i, label in enumerate(tensors):
            label_mask = torch.where(label > torch.tensor(0, dtype=torch.int8), torch.tensor(0, dtype=torch.int8), label_mask)
        tensors.append(label_mask)

        # Return the list of label tensors
        return torch.cat(tensors, 0)

    def load(self, transformer: DataTransformer):
        """
        This method checks the dimensions of the labels and the sample data

        :param transformer: the transformer that is applied to every data sample
        :param label_structure: the structure of labels to go with
        :raise ValueError: when dimensions of labels and sample don't match
        """
        # Preprocess only if that did not happen yet
        if not self.loaded:

            # Load sample
            self.sample.load(transformer=transformer)

            # Load labels
            for label in self.labels:
                label.load(transformer=transformer)

            # Remember that this sample has been checked
            self.loaded = True

    def drop(self):
        """
        TODO: implement
        """
        a = 0
