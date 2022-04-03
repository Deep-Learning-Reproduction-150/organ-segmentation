import skimage.transform as transform
from torchvision.transforms import CenterCrop
import numpy as np
import torch


class Padding(object):
    """
    Zero pads dimension until they have the desired dimensions.
    """

    def __init__(self, depth=None, width=None, height=None):
        """
        If padding for one dimension is not given, no padding will be added.

        Parameters
        ----------
        depth: int
            Desired depth dimension
        width: int
            Desired width dimension
        height: int
            Desired height dimension

        """
        self.target_depth = depth
        self.target_width = width
        self.target_height = height

    def __call__(self, img, mask=None):
        """
        Call method

        Parameters
        ----------
        img: ndarray
            Image to be padded

        Returns
        -------
        Padded image

        """
        # Get image size
        image_size = img.shape
        # Initialize paddings
        pad_right = pad_left = pad_front = pad_back = pad_top = pad_bottom = 0

        # Depth has to be padded
        if self.target_depth and image_size[2] < self.target_depth:
            pad_depth = self.target_depth - image_size[2]
            # Int casting will round down like floor
            pad_front = int(pad_depth / 2)
            pad_back = pad_depth - pad_front

        # Height has to be padded
        if self.target_height and image_size[0] < self.target_height:
            pad_height = self.target_height - image_size[0]
            pad_top = int(pad_height / 2)
            pad_bottom = pad_height - pad_top

        # Width has to be padded
        if self.target_width and image_size[1] < self.target_width:
            pad_width = self.target_width - image_size[1]
            pad_right = int(pad_width / 2)
            pad_left = pad_width - pad_right

        # Apply padding
        img = np.pad(img, ((pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right)))

        if mask is not None:
            mask = np.pad(mask, ((pad_front, pad_back), (pad_top, pad_bottom), (pad_left, pad_right)))
            return img, mask

        return torch.Tensor(img)


class Rescale(object):
    """
    Rescales a volume by a given factor
    """
    def __init__(self, depth=1, width=1, height=1):
        self.depth = depth
        self.width = width
        self.height = height

    def __call__(self, vol):
        return torch.from_numpy(transform.rescale(vol, (self.depth, self.height, self.width), anti_aliasing=True, order=3))


class Resize(object):
    """
    Resizes a volume to a desired size
    """

    def __init__(self, depth=1, width=1, height=1):
        self.depth = depth
        self.width = width
        self.height = height

    def __call__(self, vol):
        return torch.from_numpy(transform.resize(vol, (self.depth, self.height, self.width), anti_aliasing=True, order=3))


class EasyResize(object):
    """
    Resizes a volume to a desired size
    """

    def __init__(self, depth=1, width=1, height=1):
        self.depth = depth
        self.width = width
        self.height = height

    def __call__(self, vol):
        return torch.from_numpy(transform.resize(vol, (self.depth, self.height, self.width), anti_aliasing=False, order=0))

