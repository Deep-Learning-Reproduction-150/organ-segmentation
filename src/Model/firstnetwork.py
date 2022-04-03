"""
This file contains the OrganNet25D that can be trained by using the dataloader and used
by just passing instances of CTData.

Course: Deep Learning
Date: 25.03.2022
Group: 150

TODO:
    - Should we create a class Organ? So that could be outputted really nicely?
"""

import torch
from torch import nn

# from src.Dataloader.CTData import CTData


# Placeholder functions for building the model
def conv_2x2d(
    in_channels=1, out_channels=16, groups=1, kernel_size=(1, 3, 3), stride=1, padding="valid", *args, **kwargs
):
    return nn.Sequential(
        nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            groups=groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            *args,
            **kwargs,
        ),
        nn.Conv3d(
            in_channels=out_channels,
            out_channels=out_channels,
            groups=groups,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            *args,
            **kwargs,
        ),
        nn.ReLU(),  # Maybe not
    )


def conv_2x3d_coarse(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0, *args, **kwargs):
    """
    The 2xConv with 3,3,3 kernel without the ResSE presented in the paper
    """
    return nn.Sequential(
        nn.intrinsic.ConvBnReLU3d(  # does this speed things up?
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            torch.nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ),
        nn.intrinsic.ConvBnReLU3d(  # does this speed things up?
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            ),
            torch.nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ),
    )


class DoubleConvResSE(nn.Module):  # See figure 2. from the paper
    def __init__(
        self, global_pooling_size, in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0
    ) -> None:

        super().__init__()

        self.conv = conv_2x3d_coarse(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.resse = nn.Sequential(
            nn.AvgPool3d(global_pooling_size),
            nn.Flatten(),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.Sigmoid(),
            nn.Unflatten(1, (out_channels, 1, 1, 1)),
        )

    def forward(self, x):

        conv_out = self.conv(x)
        resse_out = self.resse(conv_out)

        multi = conv_out * resse_out
        y = multi + conv_out

        return y


def conv_3d_fine():
    pass


def default_pooling(kernel_size, stride):
    """
    U-Net: Convolutional Networks for Biomedical Image Segmentation reference for this.
    """
    return nn.MaxPool3d(kernel_size=kernel_size, stride=stride, padding=0, dilation=1)


class OrganNet25D(nn.Module):
    """
    This represents the OrganNet25D model as proposed by Chen et. al.

    TODO:
        - Basically implement everything :D
        - Would be really cool if this class already capsules some functionality
    """

    def __init__(self, *args, **kwargs):
        """
        Constructor method of the OrganNet
        """

        # Call torch superclass constructor
        super().__init__()

        # 2D layers
        self.two_d_1 = conv_2x2d(
            in_channels=1, out_channels=16, groups=1, kernel_size=(1, 3, 3), stride=1, padding="valid"
        )

        self.two_d_2 = conv_2x2d(
            in_channels=32, out_channels=32, groups=1, kernel_size=(1, 3, 3), stride=1, padding="valid"
        )  # TODO: Remove the padding

        # Coarse 3D layers

        # First part of 2 x Conv + ResSE
        self.coarse_3d_1 = DoubleConvResSE(
            (44, 122, 122), in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0
        )

        self.coarse_3d_2 = DoubleConvResSE(
            (18, 57, 57), in_channels=32, out_channels=64, kernel_size=(3, 3, 3), stride=1, padding=0
        )
        self.coarse_3d_3 = conv_2x3d_coarse()

        # Fine 3D block

        self.fine_3d_1 = conv_3d_fine()
        self.fine_3d_2 = conv_3d_fine()
        self.fine_3d_3 = conv_3d_fine()

        # Final 1x1x1 conv

        self.one_d_3 = nn.Conv3d(  # The final layer in the network
            in_channels=32,
            out_channels=10,
            groups=1,
            kernel_size=(1, 1, 1),
            padding=(0, 4, 4),  # TODO: Check on the padding, this is for the toy model
            *args,
            **kwargs,
        )

        return

    def forward(self, x: torch.Tensor, verbose=None, mock=True):
        """
        This method takes an image of type CTData and
        uses the model to create the segmentation of organs

        :param x: an instance of CTData
        :return: TODO: good question - what exactly?

        Input: 256,256,48
        Output: 256,256,48,10 (# organs)

        NOTE: Order unsure
        """

        if verbose:
            print("Printing shapes during the forward operation:\n")

        ## Part 1 (Left side)

        # Input: (2,1,48,256,256)

        # Input to 2D layer 1 -> Output 1 (2,16,48,252,252)
        if verbose:
            print(f"\tInput shape : {x.shape}")
        out1 = self.two_d_1(x)
        if verbose:
            print(f"\tOutput 1 shape : {out1.shape}")
        # Output 1 to max pooling layer S(1,2,2) -> Output 2 (2,16,48,126,126)
        out2 = default_pooling(kernel_size=(1, 2, 2), stride=(1, 2, 2))(out1)
        if verbose:
            print(f"\tOutput 2 shape : {out2.shape}")
        # Output 2 to coarse 3D layer 1 -> Output 3 (2,32,44,122,122)
        out3 = self.coarse_3d_1(out2)
        if verbose:
            print(f"\tOutput 3 shape : {out3.shape}")
        # Output 3 to max pooling layer S(2,2,2) -> Output 4 (2,32,22,61,61)
        out4 = default_pooling(kernel_size=(2, 2, 2), stride=(2, 2, 2))(out3)
        if verbose:
            print(f"\tOutput 4 shape : {out4.shape}")
        # Output 4 to Coarse 3D layer 2 -> Output 5 (2, 64, 18, 57, 57)
        out5 = self.coarse_3d_2(out4)
        if verbose:
            print(f"\tOutput 5 shape : {out5.shape}")
        return out5
        # Output 5 to Fine 3D Layer 1 (HDC) -> Output 6

        # Part 2 (Bottom, starting from the first Orange arrow)
        # Output 6 to Fine 3D Layer 2 -> Output 7
        # Output 7 to "Conv 1x1x1" layer 1 -> Output 8 # TODO Elaborate on the 1x1x1 layer
        # Concatenate Output 6 and Output 8 -> Output 9
        # Output 9 to Fine 3D Layer 3 -> Output 10

        # Part 3 (Right side, starting from the bottom right yellow arrow)
        # Output 10 to 1x1x1 layer 2 -> Output 11 # TODO still missing
        # Concatenate Output 11 and Output 5 -> Output 12
        # Output 12 to Coarse 3d layer 3 -> Output 13
        # Transpose Output 13 -> Output 14
        # Concatenate Output 14 and Output 3 -> Output 15
        # Output 15 to Coarse 3d Layer 4 -> Output 16
        # Concatenate Output 1 and Output 16 -> Output 17
        # Output 17 to 2D layer 2 -> Output 18
        # Output 18 to 1x1x1 layer 3 -> Final output
        out18 = x
        return out18

    def train_model(self, train_data, test_data, monitor_progress: bool = False):
        """
        This method trains the network automatically

        :param monitor_progress: if user wants detailed feedback on learning progress
        :param train_data: a data set that contains examples for training
        :param test_data: a data set that contains examples for testing
        :return: if validation was successful

        TODO:
            - How to proceed with e.g. cross validation
            - Is the paper already forcing a specific mode of operation?
            - Write validation of the trained model
        """

        # Check if system shall monitor learning progress
        if monitor_progress:

            # TODO: write some really informative progress overview
            a = 0

        # TODO: write training algorithm

        return True

    def get_organ_segments(self, x):
        """
        This method returns the actual organ segments and not raw data of the inputted
        CTData

        :param x: an example of type CTData
        :return: the organs detected in the image
        """

        # TODO: use forward method and stuff to output really nice segment representation

        return None


class ToyOrganNet25D(OrganNet25D):
    def forward(self, x, *args, **kwargs):
        out1 = self.two_d_1(x)

        # Concatenate two Output 1s to mock Output 17

        out17 = torch.cat([out1, out1], dim=1)

        out18 = self.two_d_2(out17)
        out19 = self.one_d_3(out18)

        return out19


def main():
    """
    A small toy test that the feedforward works as expected.
    """

    batch = 2
    width = height = 256
    depth = 48
    channels_in = 1
    channels_out = 10

    # Batch x Channels x Depth x Height x Width
    input_shape = (batch, channels_in, depth, height, width)
    # Batch x  Channels x Depth x Height x Width
    expected_output_shape = (batch, channels_out, depth, height, width)
    input = torch.rand(input_shape)

    model = OrganNet25D()
    # model = ToyOrganNet25D()

    output = model(input, verbose=True)

    msg = f"""
    Input shape: {input.shape}\n
    Output shape: {output.shape}\n
    Output shape correct: {output.shape == expected_output_shape} (expected: {expected_output_shape}).
    """
    print(msg)


if __name__ == "__main__":
    main()
