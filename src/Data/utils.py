"""
This file contains helpers for the data loading processes

Course: Deep Learning
Date: 03.04.2022
Group: 150
"""

import random
import torch
from src.Data.transforms import CropAroundBrainStem, GenerateSubCube
from src.utils import Logger


class DataTransformer(object):
    """
    Custom compose transform to make use of mask.
    """

    # Static list of transforms that are only allowed to be applied on the fly
    on_the_fly_transforms = [GenerateSubCube, CropAroundBrainStem]

    # Random center coordinates
    random_center = None

    # Dict containing injected organ centers
    organ_centers = None

    # If the transformation is applied while loading
    loading = None

    # Whether or whether not to apply on the fly transformations
    on_the_fly = None

    def __init__(self, transforms):
        """
        Constructor of the data transformer

        :param transforms:
        """

        # Save the transforms for later application
        self.transforms = transforms

        # For some transforms, organ centers are needed
        self.organ_centers = {}

        # Generate three random numbers for this transformer
        self.random_center = [random.random(), random.random(), random.random()]

        # Initiate on the fly and loading states
        self.on_the_fly = False
        self.loading = True

    def set_loading(self, val):
        """
        When set loading, some transforms might not be applied

        :param val:
        :return:
        """
        self.loading = val

    def set_on_the_fly(self, val):
        """
        Set in on fly mode and also apply on the fly transformations

        :param val:
        :return:
        """
        self.on_the_fly = val

    def inject_organ_center(self, organ: str, center):
        self.organ_centers[organ] = center

    def __call__(self, img):

        # Check if transformation is applied on batch of samples
        if len(img.size()) == 4:

            subs = []
            for channel in range(img.size()[0]):
                subs.append(self(img[channel, ...]).unsqueeze(0))
            img = torch.cat(subs, 0)

        # Transformation is applied on single CT Image
        else:

            for t in self.transforms:

                # Check if this transformation is a on the fly transform
                if (type(t) in DataTransformer.on_the_fly_transforms):

                    # Check if on the fly transforms also shall be applied
                    if self.on_the_fly:

                        # Check if transform is generate cube transformation
                        if type(t) is GenerateSubCube:
                            img = t(img, self.random_center)

                        # Check if transform is CropAroundBrainStem transformation
                        if type(t) is CropAroundBrainStem:
                            if 'BrainStem' not in self.organ_centers:
                                Logger.log("The transformation CropAroundBrainStem can not be applied")
                            img = t(img, self.organ_centers['BrainStem'])

                # Check if loading transforms should also be applied
                elif self.loading:
                    img = t(img)

        return img
