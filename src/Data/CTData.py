"""
This file contains the CTData class representing a CT dataset

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

import os
import torch
import nrrd
import imageio
import matplotlib.pyplot as plt
from torch import from_numpy
from src.utils import bcolors, Logger


class CTData:
    """
    This class represents a CT Image and is used to depict HaN CT images
    """

    # This attribute stores the data in a ndarray format
    data = None

    # Path where the data file is stored
    path = None

    # The name of the file (used for labels as well)
    name = None

    # This meta data contains information about the data obtained from the input file
    meta = None

    # Whether the data is loaded in this object
    loaded = None

    def __init__(self, **params):
        """
        Constructor of a CT Image

        :param path: the path where the nrrd file is
        :param preload: whether to load data directly when creating
        """

        # Get path and data from params
        path = params.get('path', None)
        data = params.get('data', None)
        name = params.get('name', None)

        # Save the path of the datafile
        if path is not None:

            # Check if this file exists
            if not os.path.exists(path):
                # Raise an exception for this issue
                raise ValueError(bcolors.FAIL + "ERROR: Given path does not lead to a nrrd file" + bcolors.ENDC)

            # Save path for later operations
            self.path = path

            # Initiate header and data
            self.data = None
            self.meta = None

            # Extract the name of the file from the path where it is located
            if name is None:
                path, file = os.path.split(self.path)
                self.name = file
            else:
                self.name = name

        if data is not None:

            # Save data already
            self.data = data
            self.name = name
            self.loaded = True

    def drop(self):
        # TODO: implement dropping of data
        a = 0

    def load(self, transformer=None):
        """
        This method reads the data and can apply a transformation to it before storing it

        :param transformer: a data transformer to apply to the data while reading it
        """

        # Check if data has been loaded already
        if not self.loaded:

            # Try to load the data at the given path
            try:

                # Load the data and throw it into an ndarray
                extracted_data, header = nrrd.read(self.path)

                # Save the as attributes for this instance
                self.data = from_numpy(extracted_data).to(torch.float32)
                self.meta = header

                # Check if the data has three dimensions
                if self.data.ndim != 3:
                    raise ValueError(bcolors.FAIL + "ERROR: Unexpected number of dimensions (" + str(
                        self.data.ndim) + ") in data sample" + bcolors.ENDC)

                # Remember that the data has been loaded
                self.loaded = True

            except Exception as error:

                # Raise exception that file could not be loaded
                raise ValueError(
                    bcolors.FAIL + "ERROR: could not read nrrd file at " + self.path + "(" + str(error) + ")" + bcolors.ENDC)

        # Apply transformations
        if transformer is not None:

            # Apply the transformer to the data in place
            self.data = transformer(self.data)

    def get_tensor(self, transformer=None):
        """
        This method returns the tensor containing the data. If a transformer is passed, the data is transformed on the fly.

        :param transformer: a data transformer that is applied to data before output
        :return data: (transformed) tensor
        """

        # Check if preloaded or have to load now
        if not self.loaded:

            # Load data from the file
            self.load()

        # Check if transformer has been passed
        if transformer is not None:

            # Return the transformed data
            return transformer(self.data)

        # Return a tensor of data
        return self.data

    def visualize(self, show: bool = False, export_png: bool = False, export_gif: bool = False,
                  direction: str = "vertical", name: str = None, high_quality: bool = False,
                  show_status_bar: bool = True):
        """

        Visualize the data using matplotlib

        :param show: directly displays the images here
        :param name: either None (default name) or special name for file
        :param high_quality: if True, HQ images are going to be exported (about 50MB / GIF)
        :param direction: how to go through image, options: vertical, horizontal
        :param export_png: whether system shall create png images for the slices
        :param export_gif: whether system shall create a GIF file from the data
        :param show_status_bar: progress bar will be displayed to show progress of generation
        """

        # Check if preloaded or have to load now
        if self.data is None:
            # Load data from the file
            self.load()

        # Check given parameters
        if direction not in ['vertical', 'horizontal']:
            raise ValueError(bcolors.FAIL + "ERROR: Direction has to either be 'vertical' or 'horizontal'" + bcolors.ENDC)

        # Print a status update
        Logger.log("Creating visualization of " + str(self.data.ndim) + "-dimensional data " + self.name + " with direction " + direction, in_cli=True)

        # Extract the three dimensions from the data set
        shape = self.data.shape
        x_dimensions = shape[0]
        z_dimensions = shape[2]
        dim_counter = z_dimensions if direction == 'vertical' else x_dimensions

        # Filenames
        images = []

        # Iterate through all layers of the image
        for index in range(dim_counter):

            # Get the data from this layer
            slice_data = self.data[:, :, index] if direction == 'vertical' else self.data[index, :, :]

            # Create figure size tupel depending on quality and direction
            if high_quality:
                figsize = (14, 14) if direction == 'vertical' else (8, 15)
            else:
                figsize = (9.8, 9.8) if direction == 'vertical' else (5.6, 10.5)

            # Create an image
            plt.figure(figsize=figsize)
            plt.gray()
            plt.imshow(slice_data)
            plt.draw()

            # Print additional status updates
            plt.title(name + ', ' + direction + ' (slice ' + str(index) + ')')
            plt.xlabel("X Direction" if direction == "vertical" else "Depth (z)")
            plt.ylabel("Y Direction")

            if show:
                plt.show()

            # If export png is on, save export
            if export_png:

                # Check if the folder exists
                if not os.path.isdir('visualizations'):
                    # Create folder as it does not exist yet
                    os.mkdir('visualizations')

                # Folder name for the png output
                folder_name = 'visualizations/' + (self.name if name is None else name)

                # Check if the folder exists
                if not os.path.isdir(folder_name):
                    # Create folder as it does not exist yet
                    os.mkdir(folder_name)

                # Create a file for that image
                plt.savefig(folder_name + '/slice_' + str(index) + '.png', dpi=100)

            # Append this
            if export_gif:

                # Check if the folder exists
                if not os.path.isdir('visualizations'):
                    # Create folder as it does not exist yet
                    os.mkdir('visualizations')

                tmp_image_path = 'visualizations/tmp.png'
                plt.savefig(tmp_image_path, dpi=100)
                images.append(imageio.imread(tmp_image_path))

            # Close the image
            plt.close()

            # Print the changing import status line
            if show_status_bar:
                done = ((index + 1) / dim_counter) * 100
                Logger.print_status_bar(done=done, title="processing")

            # Always stop status bar after this
            if show_status_bar:
                Logger.end_status_bar()

        # If system shall export a GIF from it, do so
        if export_gif:

            # Print status update
            Logger.log("Creating visualization of " + str(self.data.ndim) + "-dimensional data " + str(self.name) +
                       ", saving GIF file", in_cli=True)

            # Remove the tmp tile
            os.remove('visualizations/tmp.png')

            # Save GIF file
            imageio.mimsave('visualizations/' + (self.name if name is None else name) + '.gif', images)