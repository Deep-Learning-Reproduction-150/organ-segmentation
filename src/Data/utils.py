"""
This file contains helpers for the data loading processes

Course: Deep Learning
Date: 03.04.2022
Group: 150
"""

from src.Data.transforms import CropAroundBrainStem
from src.utils import Logger


class DataTransformer(object):
    """
    Custom compose transform to make use of mask.
    """

    def __init__(self, transforms):
        self.transforms = transforms
        self.organ_centers = {}

    def inject_organ_center(self, organ: str, center):
        self.organ_centers[organ] = center

    def __call__(self, img):
        for t in self.transforms:
            a = 0
            # Check if transform is the special brain stem transformation
            if type(t) is CropAroundBrainStem:
                if 'BrainStem' not in self.organ_centers:
                    Logger.log("The transformation CropAroundBrainStem can not be applied")
                img = t(img, self.organ_centers['BrainStem'])
            else:
                img = t(img)
        return img
