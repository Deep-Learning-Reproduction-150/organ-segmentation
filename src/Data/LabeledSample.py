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
import json
from medpy.io import load
import torch
from src.utils import bcolors, Logger
from src.Data.CTData import CTData
import numpy as np
from scipy import ndimage
from src.Data.utils import DataTransformer
from src.Data.transforms import CropAroundBrainStem


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

    # Attribute storing the brain stem center
    brain_stem_center = None

    def __init__(self,
                 path,
                 sample_transformer: DataTransformer = DataTransformer([]),
                 label_transformer: DataTransformer = DataTransformer([])
                 ):
        """
        Constructor of the LabeledSample object. Expected by default is a folder that contains one nrrd file which
        is the sample data and a folder with name <labels_folder_path> that contains n labels, itself encoded as nrrd
        files.

        :param path: the path to the folder that contains the files
        """

        # Assign an id and increment the id store
        self.id = LabeledSample.id_store
        LabeledSample.id_store += 1

        self.label_transformer = label_transformer
        self.sample_transformer = sample_transformer

        # Initiate loaded with false
        self.loaded = False

        # Initialize the brain stem center with none
        self.brain_stem_center = None

        # Save the path to this sample
        self.path = path

        # Check if this file exists
        if not os.path.isdir(path):
            # Raise an exception for this issue
            raise ValueError(bcolors.FAIL + "ERROR: Given path does not lead to a folder" + bcolors.ENDC)

        # Check whether mha or nrrd files
        if "sample" in os.path.split(path)[-1]:
            self._read_transformed_tensors(path)
        elif os.path.isfile(os.path.join(path, 'voxelinfo.json')):
            self._read_mha(path)
        else:
            self._read_nrrd(path)

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

        # Inject transformer insights
        self.sample_transformer.inject_organ_center('BrainStem', self._get_brain_stem_center())
        self.sample_transformer.output_mode = True

        # Get the tensor with the transformers
        tensor = self.sample.get_tensor(transformer=self.sample_transformer)

        # TODO: random picking of "subcube"

        # Return the sample (which is a tensor)
        return tensor

    def get_labels(self, label_structure: list):
        """
        This method returns the list of labels associated with this sample

        :return labels: list of tensors that are the labels
        """

        self.label_transformer.inject_organ_center('BrainStem', self._get_brain_stem_center())
        self.label_transformer.output_mode = True

        # Check whether the passed label order is there
        if len(label_structure) == 0:

            # Stop the process as the order of labels is unknown
            raise ValueError("Labeled data object received an empty label order list, stopping")

        # Initialize a list of labels
        tensors = []

        # Iterate through the labels and create
        for wanted_label in label_structure:

            # Iterate through the labels and find it
            data = None
            for label in self.labels:
                if label.name == wanted_label:
                    data = label.get_tensor(transformer=self.label_transformer, dtype=torch.int8)
                    break

            # Check if label exists
            if data is None:
                # Create zero sample
                if len(tensors) > 0:
                    # Use an existing label tensor as reference
                    data = torch.zeros_like(tensors[0], dtype=torch.int8)
                else:
                    # Have to obtain sample tensor (much more inefficient)
                    data = torch.zeros_like(self.get_tensor(), dtype=torch.int8)

            # Append the transformed label to it
            tensors.append(data)

        # Compute a "background label" and append it to the labels
        if len(tensors) > 0:

            # Create background slice
            label_mask = torch.ones_like(tensors[0], dtype=torch.int8)
            for i, label in enumerate(tensors):
                label_mask = torch.where(label > torch.tensor(0, dtype=torch.int8), torch.tensor(0, dtype=torch.int8), label_mask)
            tensors.append(label_mask)

            # Return the list of label tensors
            return torch.cat(tensors, 0)
        else:

            # Warn about no tensors
            raise ValueError("Labeled data object does not contain any labels, stopping")

    def load(self):
        """
        This method checks the dimensions of the labels and the sample data

        :param sample_transformer: transformer for the sample
        :param label_transformer: transformer for the labels
        :raise ValueError: when dimensions of labels and sample don't match
        """

        # Preprocess only if that did not happen yet
        if not self.loaded:

            # Load sample
            self.sample.load(transformer=self.sample_transformer)

            # Load labels
            for label in self.labels:
                label.load(transformer=self.label_transformer)

            # Remember that this sample has been checked
            self.loaded = True

    def drop(self):
        """
        TODO: implement
        """
        a = 0

    def _get_brain_stem_center(self):
        """
        This function computes the brain stem center for this sample and saves and returns it

        :return: 3D center of brain stem
        """
        if self.brain_stem_center is None:
            for label in self.labels:
                if label.name == 'BrainStem':
                    bs = label.get_tensor()
                    center_of_gravity = ndimage.center_of_mass(np.array((bs.transpose(1, -1))[0, :, :, :]))
                    self.brain_stem_center = center_of_gravity
                    return self.brain_stem_center
            Logger.log("Brain stem not contained in data sample " + str(self.id), type="ERROR", in_cli=True)
            return None
        else:
            return self.brain_stem_center

    def _read_transformed_tensors(self, path):
        # Check if everything is there
        if not os.path.isdir(os.path.join(path, 'labels')):
            raise ValueError("Transformed dataset has wrong structure. Delete and re-create!")
        self.labels = []
        for element in glob.glob(os.path.join(path, 'labels') + "/*.pt"):
            # Create a label for storing
            label = CTData(path=element)
            self.labels.append(label)
        sample_path = os.path.join(path, 'sample.pt')
        self.sample = CTData(path=sample_path)

    def _read_nrrd(self, path):
        # Check if this file exists
        if not os.path.isdir(os.path.join(path, 'structures')):
            # Raise an exception for this issue
            raise ValueError(bcolors.FAIL + "ERROR: labels are not found (looking for ./structures)" + bcolors.ENDC)

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
        for element in glob.glob(os.path.join(path, 'structures') + "/*.nrrd"):
            # Create a label for storing
            label = CTData(path=element)
            # Store the label in the labels attribute
            self.labels.append(label)

    def _read_mha(self, path):

        # Open the config file and load voxel description
        with open(os.path.join(path, 'voxelinfo.json'), 'r') as f:

            # Load the voxel description
            voxel_description = json.load(f)

            # Load the organ configuration
            if 'resampled' in voxel_description and 'labels_oars' in voxel_description['resampled']:

                # Load the organs at risk in the masks
                oars = voxel_description['resampled']['labels_oars']

                # Load the data from the mha files
                mask_data_path = 'mask_resampled_' + os.path.split(path)[-1] + '.mha'
                image_data_path = 'img_resampled_' + os.path.split(path)[-1] + '.mha'
                mask_data = torch.from_numpy(load(os.path.join(path, mask_data_path))[0])

                # Initiate labels
                self.labels = []

                # Compute the center of gravity TODO: this is ugly - preprocessing must be rebuilt
                for i, organ in enumerate(oars):
                    if organ == 'BrainStem':
                        brain_stem_tensor = torch.where(mask_data == i, 1, 0)
                        for t in self.label_transformer.transforms:
                            if type(t) == CropAroundBrainStem:
                                break
                            brain_stem_tensor = t(brain_stem_tensor)
                        center_of_gravity = ndimage.center_of_mass(np.array(brain_stem_tensor))
                        self.brain_stem_center = center_of_gravity

                        # Inject to transformers
                        self.sample_transformer.inject_organ_center('BrainStem', self._get_brain_stem_center())
                        self.label_transformer.inject_organ_center('BrainStem', self._get_brain_stem_center())

                        break

                # Create label mask CTData instances
                for i, organ in enumerate(oars):

                    # Skip background (to stick to existing logic)
                    if i != 0:

                        # Create a label for storing
                        label = CTData(name=organ, path=os.path.join(self.path, mask_data_path), channel_index=i)
                        # Store the label in the labels attribute
                        self.labels.append(label)

                # Create the sample CT file instance
                self.sample = CTData(path=os.path.join(self.path, image_data_path), name="img")

            else:

                Logger.log("The voxelinfo.json file did not contain the expected information", type="ERROR",
                           in_cli=True)

    def _read_mha_directly(self, path):

        # Open the config file and load voxel description
        with open(os.path.join(path, 'voxelinfo.json'), 'r') as f:

            # Load the voxel description
            voxel_description = json.load(f)

            # Load the organ configuration
            if 'resampled' in voxel_description and 'labels_oars' in voxel_description['resampled']:

                # Load the organs at risk in the masks
                oars = voxel_description['resampled']['labels_oars']

                # Load the data from the mha files
                mask_data_path = 'mask_resampled_' + os.path.split(path)[-1] + '.mha'
                image_data_path = 'img_resampled_' + os.path.split(path)[-1] + '.mha'
                mask_data = torch.from_numpy(load(os.path.join(path, mask_data_path))[0])
                img_data = torch.from_numpy(load(os.path.join(path, image_data_path))[0]).to(torch.float32)

                # Initiate labels
                self.labels = []

                # Compute the center of gravity TODO: this is ugly - preprocessing must be rebuilt
                for i, organ in enumerate(oars):
                    if organ == 'BrainStem':
                        brain_stem_tensor = torch.where(mask_data == i, 1, 0)
                        for t in self.label_transformer.transforms:
                            if type(t) == CropAroundBrainStem:
                                break
                            brain_stem_tensor = t(brain_stem_tensor)
                        center_of_gravity = ndimage.center_of_mass(np.array(brain_stem_tensor))
                        self.brain_stem_center = center_of_gravity

                        # Inject to transformers
                        self.sample_transformer.inject_organ_center('BrainStem', self._get_brain_stem_center())
                        self.label_transformer.inject_organ_center('BrainStem', self._get_brain_stem_center())

                        break

                # Create label mask CTData instances
                for i, organ in enumerate(oars):

                    # Skip background (to stick to existing logic)
                    if i != 0:

                        # Compose label tensors
                        zero_mask = torch.zeros_like(mask_data)
                        organ_mask = torch.where(mask_data == i, torch.tensor(1).to(torch.uint8), zero_mask)

                        # Create a label for storing
                        label = CTData(data=self.label_transformer(organ_mask), name=organ, loaded=True)
                        # Store the label in the labels attribute
                        self.labels.append(label)

                # Create the sample CT file instance
                self.sample = CTData(data=self.sample_transformer(img_data), name="img", loaded=True)

                # Set loaded true
                self.loaded = True

            else:

                Logger.log("The voxelinfo.json file did not contain the expected information", type="ERROR", in_cli=True)
