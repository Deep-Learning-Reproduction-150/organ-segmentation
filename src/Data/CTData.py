"""
This file contains the CTData class representing a CT dataset

Course: Deep Learning
Date: 25.03.2022
Group: 150
"""

import os
import nrrd
import imageio
import matplotlib.pyplot as plt
from torch import from_numpy
from src.utils import bcolors, Logger


class CTData:
    """
    This class represents a CT Image and is used to depict HaN CT images

    TODO:
        - What "else" functionality should a CT Image have?
    """

    # This attribute stores the data in a ndarray format
    data = None

    # Path where the data file is stored
    path = None

    # The name of the file (used for labels as well)
    name = None

    # This meta data contains information about the data obtained from the input file
    meta = None

    # Whether to preload data
    preload = None

    def __init__(self, path: str, preload: bool = True):
        """
        Constructor of a CT Image

        :param path: the path where the nrrd file is
        :param preload: whether to load data directly when creating
        """

        # Save whether data should be preloaded
        self.preload = preload

        # Save the path of the datafile
        self.path = path

        # Check if this file exists
        if not os.path.exists(path):
            # Raise an exception for this issue
            raise ValueError(bcolors.FAIL + "ERROR: Given path does not lead to a nrrd file" + bcolors.ENDC)

        # Initiate header and data
        self.data = None
        self.meta = None

        # If preload, load data now
        if self.preload:

            # Load data from the file
            self._load_data_from_file()

        # Extract the name of the file from the path where it is located
        filename = path.split('/')[-1]
        self.name = filename.split('.')[0]

    def _load_data_from_file(self):
        """
        This method read the data
        """

        # Try to load the data at the given path
        try:

            # Load the data and throw it into an ndarray
            extracted_data, header = nrrd.read(self.path)

            # Save the as attributes for this instance
            self.data = extracted_data
            self.meta = header

            # Check if the data has three dimensions
            if self.data.ndim != 3:
                raise ValueError(bcolors.FAIL + "ERROR: Unexpected number of dimensions (" + str(
                    self.data.ndim) + ") in data sample" + bcolors.ENDC)

            # Check if data dimensions are correct
            if self.meta['dimension'] != 3:
                raise ValueError(bcolors.FAIL + "ERROR: file " + self.path + " contains " + str(
                    self.meta['dimension']) + "-dimensional data (not expected 3D data)" + bcolors.ENDC)

        except Exception as error:

            # Raise exception that file could not be loaded
            raise ValueError(
                bcolors.FAIL + "ERROR: could not read nrrd file at " + self.path + "(" + str(error) + ")" + bcolors.ENDC)

    def get_tensor(self):
        """
        This method returns a three dimensional ndarray that contains the data

        :return data: raw ndarray data
        """

        # Check if preloaded or have to load now
        if self.data is None:

            # Load data from the file
            self._load_data_from_file()

        # Return a tensor of data
        return from_numpy(self.data)

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
            self._load_data_from_file()

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