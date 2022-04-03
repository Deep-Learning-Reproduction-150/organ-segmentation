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

    # This attribute stores the CTData
    sample = None
    transformed_sample = None

    # This list stores the labels (also of type CTData)
    labels = None
    transformed_labels = None

    # Attribute that stores the path to the folder that contains the sample data
    path = None

    # Whether or whether not to preload data
    preload = None

    # Stores whether the sample has been processed already
    preprocessed = None

    def __init__(self, path, preload: bool = True, labels_folder_path: str = "structures"):
        """
        Constructor of the LabeledSample object. Expected by default is a folder that contains one nrrd file which
        is the sample data and a folder with name <labels_folder_path> that contains n labels, itself encoded as nrrd
        files.

        :param path: the path to the folder that contains the files
        :param preload: whether to load the data directly when instantiating an object
        :param labels_folder_path: folder within path that contains files with labels

        TODO: make sure that every instance of LabeledSample has the same label structure!!!
        """

        # Save whether sample should reload data
        self.preload = preload

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
            self.sample = CTData(glob.glob(path + "/*.nrrd")[0], preload=self.preload)

        # Initiate a sample list
        self.labels = []

        # Iterate through the labels and create CT image instances for them as well
        for element in glob.glob(os.path.join(path, labels_folder_path) + "/*.nrrd"):
            # Create a label for storing
            label = CTData(element, preload=self.preload)
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

    def get_tensor(self, take_original: bool = False):
        """
        This method returns a tensor that contains the data of this sample

        :return tensor: which contains the data points
        """

        # Check if sample has been preprocessed
        if not self.preprocessed:
            raise Exception("ERROR: Data sample has not been preprocessed yet")

        # Return the sample (which is a tensor)
        return self.sample.get_tensor() if take_original else self.transformed_sample

    def get_labels(self):
        """
        This method returns the list of labels associated with this sample

        :return labels: list of tensors that are the labels
        """

        # Check if sample has been preprocessed
        if not self.preprocessed:
            raise Exception("ERROR: Data sample has not been preprocessed yet")

        # Initialize a list of labels
        tensors = []
        labels = []

        # Iterate through the labels and get tensors of each
        for label in self.labels:
            # Append a tensor
            tensors.append(label.get_tensor())
            labels.append(label.name)

        # TODO: go veikko
        label_threshold = -1000.0
        background_voxel_value = -1024.0

        label_mask = torch.zeros_like(self.sample)  # By default choose entire image

        for label in labels:
            current_label_mask = label > label_threshold  # Choose volume under the organ
            label_mask = label_mask | current_label_mask  # Select it

        background = deepcopy(self.sample)
        background[~label_mask] = background_voxel_value
        tensors.append(background)
        labels.append("background")

        # return label data
        label_data = {"features": tensors, "label": labels}

        # Return the list of labels
        return label_data

    def preprocess(self, transformer: DataTransformer, label_structure: list, output_info: bool = False):
        """
        This method checks the dimensions of the labels and the sample data

        :param transformer: the transformer that is applied to every data sample
        :param label_structure: the structure of labels to go with
        :raise ValueError: when dimensions of labels and sample don't match
        """
        # Preprocess only if that did not happen yet
        if not self.preprocessed:

            # If outputting of info should happen, do it here
            if output_info:
                Logger.log("Preprocessing and applying transformations to sample " + str(self.id))

            # Get the tensor of the sample CT data
            sample_tensor = self.sample.get_tensor()

            # Iterate through all labels
            for label in self.labels:

                # Get the tensor of the label
                label_tensor = label.get_tensor()

                # Check if any dimension mismatches (no of dimensions is checked in CTData)
                x_mismatch = label_tensor.data.shape[0] != sample_tensor.data.shape[0]
                y_mismatch = label_tensor.data.shape[1] != sample_tensor.data.shape[1]
                z_mismatch = label_tensor.data.shape[2] != sample_tensor.data.shape[2]
                if x_mismatch or y_mismatch or z_mismatch:
                    Logger.log(
                        "Dimension of LabeledSample at "
                        + str(self.path)
                        + " do not match: "
                        + str(sample_tensor.data.shape)
                        + " VS. "
                        + str(label_tensor.data.shape),
                        type="ERROR",
                        in_cli=True,
                    )

            # Initiate transformed labels
            self.transformed_labels = []

            # Transform the sample data using the passed transformer
            self.transformed_sample = transformer(sample_tensor.transpose(0, -1))

            # Iterate through the labels and create
            for wanted_label in label_structure:

                # Iterate through the labels and find it
                label = None
                match = False
                for label in self.labels:
                    if label.name == wanted_label:
                        match = True
                        break

                # Check if label exists
                if match and label is not None:
                    data = transformer(label.get_tensor().transpose(0, -1)).unsqueeze(0)
                else:
                    # Create zero sample
                    data = torch.zeros(self.transformed_sample.size())
                    data = transformer(data).unsqueeze(0)

                # Append the transformed label to it
                self.transformed_labels.append(data)

        # Remember that this sample has been checked
        self.preprocessed = True
