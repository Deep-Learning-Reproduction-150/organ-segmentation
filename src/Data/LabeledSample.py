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

    def get_tensor(self):
        """
        This method returns a tensor that contains the data of this sample

        :return tensor: which contains the data points
        """

        # Check if sample has been preprocessed
        if not self.loaded:
            raise Exception("ERROR: Data sample has not been preprocessed yet")

        # Return the sample (which is a tensor)
        return self.sample.get_tensor()

    def get_labels(self):
        """
        This method returns the list of labels associated with this sample

        :return labels: list of tensors that are the labels
        TODO: not used?
        """

        # Check if sample has been preprocessed
        if not self.loaded:
            raise Exception("ERROR: Data sample has not been preprocessed yet")

        # Initialize a list of labels
        tensors = []
        labels = []

        # Iterate through the labels and get tensors of each
        for label in self.labels:
            # Append a tensor
            tensors.append(label.get_tensor())
            labels.append(label.name)

        # return label data
        label_data = {"features": tensors, "label": labels}

        # Return the list of labels
        return label_data

    def load(self, transformer: DataTransformer, label_structure: list):
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

            # Get the transformed tensor from the sample
            transformed_sample = self.sample.get_tensor()

            # Change the depth and x dimension
            self.sample = transformed_sample.unsqueeze(0)

            # Initiate transformed labels
            transformed_labels = []

            # Iterate through the labels and create
            for wanted_label in label_structure:

                # Iterate through the labels and find it
                data = None
                for label in self.labels:
                    if label.name == wanted_label:
                        data = label.get_tensor().unsqueeze(0)
                        break

                # Check if label exists
                if data is None:
                    # Create zero sample
                    data = torch.zeros(transformed_sample.size())
                    data = transformer(data).unsqueeze(0)

                # Append the transformed label to it
                transformed_labels.append(data)

            # Replace the labels with transformed ones
            self.labels = transformed_labels

            # TODO: The following could maybe get some more improvements

            # By default choose entire image
            label_mask = torch.zeros_like(self.sample, dtype=torch.bool)
            background_voxel_value = self.sample.min()

            # Iterate through the transformed labels
            for label in self.labels:
                label_threshold = label.mean()
                current_label_mask = label > label_threshold  # Choose volume under the organ
                label_mask = label_mask | current_label_mask  # Select it

            background = deepcopy(self.sample)
            background[label_mask] = background_voxel_value
            self.labels.append(background)

        # Remember that this sample has been checked
        self.loaded = True

    def drop(self):
        a = 0
