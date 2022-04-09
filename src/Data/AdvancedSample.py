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
import torch
import nrrd
from medpy.io import load
from src.utils import bcolors, Logger
from src.Data.CTData import CTData
import numpy as np
from scipy import ndimage
from src.Data.utils import DataTransformer


class AdvancedSample:
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
    label_structure = None

    # Attribute that stores the path to the folder that contains the sample data
    path = None

    # Stores whether the sample has been processed already
    loaded = None

    # Attribute storing the brain stem center
    brain_stem_center = None

    def __init__(self,
                 path,
                 label_structure: list,
                 sample_transformer: DataTransformer = DataTransformer([]),
                 label_transformer: DataTransformer = DataTransformer([]),
                 ):
        """
        Constructor of the LabeledSample object. Expected by default is a folder that contains one nrrd file which
        is the sample data and a folder with name <labels_folder_path> that contains n labels, itself encoded as nrrd
        files.

        :param path: the path to the folder that contains the files
        """

        # Assign an id and increment the id store
        self.id = AdvancedSample.id_store
        AdvancedSample.id_store += 1

        # Save the transformers
        self.label_transformer = label_transformer
        self.sample_transformer = sample_transformer

        # Initiate loaded with false
        self.loaded = False

        # Save the wanted label structure
        self.label_structure = label_structure

        # Initialize the brain stem center with none
        self.brain_stem_center = None

        # Save the path to this sample
        self.path = path

        self.sample = None
        self.labels = None

        # Check if this file exists
        if not os.path.isdir(path):
            # Raise an exception for this issue
            raise ValueError(bcolors.FAIL + "ERROR: Given path does not lead to a folder" + bcolors.ENDC)

    def get_tensor(self):
        """
        This method returns a tensor that contains the data of this sample

        :return tensor: which contains the data points
        """

        # Inject transformer insights
        self.sample_transformer.inject_organ_center('BrainStem', self._get_brain_stem_center())
        self.sample_transformer.set_on_the_fly(True)

        if self.loaded:
            tensor = self.sample_transformer(self.sample)
        else:
            tensor = self.sample_transformer(self._get_sample_from_file())

        # Return the sample (which is a tensor)
        return tensor.unsqueeze(0)

    def get_labels(self):
        """
        This method returns the list of labels associated with this sample

        :return labels: list of tensors that are the labels
        """

        # Inject transformer insights
        self.label_transformer.inject_organ_center('BrainStem', self._get_brain_stem_center())
        self.label_transformer.set_on_the_fly(True)

        if self.loaded:
            tensor = self.label_transformer(self.labels)
        else:
            tensor = self.label_transformer(self._get_labels_from_file())

        # Return the tensor with batch dimension
        return tensor.unsqueeze(0)

    def load(self):
        """
        This method checks the dimensions of the labels and the sample data

        :raise ValueError: when dimensions of labels and sample don't match
        """

        # Preprocess only if that did not happen yet
        if not self.loaded:

            # Set the data transformer in loading state
            self.sample_transformer.set_loading(True)
            self.label_transformer.set_loading(True)

            # Load sample
            self.sample = self.sample_transformer(self._get_sample_from_file())

            # Load labels
            self.labels = self.label_transformer(self._get_labels_from_file())

            # Remember that this sample has been checked
            self.loaded = True

            # Set the data transformer in loading state
            self.sample_transformer.set_loading(False)
            self.label_transformer.set_loading(False)

    def _get_brain_stem_center(self):
        """
        This function computes the brain stem center for this sample and saves and returns it

        :return: 3D center of brain stem
        """
        if self.brain_stem_center is None:

            bs_index = self.label_structure.index("BrainStem")
            labels = self.labels if self.loaded else self.label_transformer(self._get_labels_from_file())
            bs_mask = labels[bs_index, ...]
            center_of_gravity = ndimage.center_of_mass(np.array(bs_mask))
            self.brain_stem_center = center_of_gravity
            return self.brain_stem_center
        else:
            return self.brain_stem_center

    def _get_labels_from_file(self):

        # Check whether mha or nrrd files
        if os.path.isfile(os.path.join(self.path, 'voxelinfo.json')):
            # Open the config file and load voxel description
            with open(os.path.join(self.path, 'voxelinfo.json'), 'r') as f:

                # Load the voxel description
                voxel_description = json.load(f)

                # Load the organ configuration
                if 'resampled' in voxel_description and 'labels_oars' in voxel_description['resampled']:
                    # Load the organs at risk in the masks
                    oars = voxel_description['resampled']['labels_oars']
                    # Load the data from the mha files
                    mask_data_path = 'mask_resampled_' + os.path.split(self.path)[-1] + '.mha'
                    data_tensor = torch.from_numpy(load(os.path.join(self.path, mask_data_path))[0]).to(torch.float32)
                    tensor = torch.zeros([len(self.label_structure)] + list(data_tensor.size()))
                    for i, organ in enumerate(self.label_structure):
                        if organ in oars:
                            index = oars.index(organ)
                            tensor[i, ...] = torch.where(data_tensor == index, 1., 0.)
                    return tensor

        else:
            raise ValueError("nrrd files are not supported at the moment")
            label_tensors = []
            # Iterate through the labels and create CT image instances for them as well
            for element in glob.glob(os.path.join(self.path, 'structures') + "/*.nrrd"):
                extracted_data, header = nrrd.read(element)
                label_tensors.append(extracted_data)
            label_tensor = torch.cat(label_tensors, dim=0)
            # TODO: some wild transpose to get the order right
            return label_tensor

    def _get_sample_from_file(self):

        # Check whether mha or nrrd files
        if os.path.isfile(os.path.join(self.path, 'voxelinfo.json')):
            # Load mha data
            image_data_path = 'img_resampled_' + os.path.split(self.path)[-1] + '.mha'
            data_tensor = torch.from_numpy(load(os.path.join(self.path, image_data_path))[0]).to(torch.float32)
            return data_tensor
        else:
            raise ValueError("nrrd files are not supported at the moment")
            # Check if the folder is encoded in the expected format
            if len(glob.glob(self.path + "/*.nrrd")) > 1:
                # Print error that more then one data file was found
                raise Exception("ERROR: more than one sample data file found during creation of LabeledSample")
            else:
                # Load the data and throw it into an ndarray
                extracted_data, header = nrrd.read(self.path)
                # Save the as attributes for this instance
                return torch.from_numpy(extracted_data).to(torch.float32)
