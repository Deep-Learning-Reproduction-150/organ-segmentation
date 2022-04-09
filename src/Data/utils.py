"""
This file contains helpers for the data loading processes

Course: Deep Learning
Date: 03.04.2022
Group: 150
"""

import random
from src.Data.transforms import CropAroundBrainStem, GenerateSubCube
from src.utils import Logger


class DataTransformer(object):
    """
    Custom compose transform to make use of mask.
    """

    # Random center coordinates
    random_center = None

    # Dict containing injected organ centers
    organ_centers = None

    output_mode = None

    def __init__(self, transforms):
        """
        Constructor of the data transformer

        :param transforms:
        """

        self.transforms = transforms

        self.output_mode = False

        # For some transforms, organ centers are needed
        self.organ_centers = {}
        # Generate three random numbers for this transformer
        self.random_center = [random.random(), random.random(), random.random()]

    def inject_organ_center(self, organ: str, center):
        self.organ_centers[organ] = center

    def __call__(self, img):
        for t in self.transforms:
            # Check if transform is the special brain stem transformation
            if type(t) is GenerateSubCube:
                img = t(img, self.random_center)
            elif type(t) is CropAroundBrainStem:
                if self.output_mode:
                    if 'BrainStem' not in self.organ_centers:
                        Logger.log("The transformation CropAroundBrainStem can not be applied")
                    img = t(img, self.organ_centers['BrainStem'])
            else:
                img = t(img)
        return img
