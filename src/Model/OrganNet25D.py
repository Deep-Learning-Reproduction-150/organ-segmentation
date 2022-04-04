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

from .utils import conv_2x3d_coarse, HDC, conv_2x2d, crop3d

# from src.Dataloader.CTData import CTData


class DoubleConvResSE(nn.Module):  # See figure 2. from the paper
    def __init__(
        self,
        global_pooling_size,
        activation=nn.Sigmoid(),
        in_channels=16,
        out_channels=32,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=0,
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
            activation,
            nn.Unflatten(1, (out_channels, 1, 1, 1)),
        )

    def forward(self, x):

        conv_out = self.conv(x)
        resse_out = self.resse(conv_out)

        multi = conv_out * resse_out
        y = multi + conv_out

        return y


class HDCResSE(nn.Module):  # See figure 2. from the paper
    def __init__(
        self,
        global_pooling_size,
        dilations=(1, 2, 5),
        in_channels=16,
        out_channels=32,
        kernel_size=(3, 3, 3),
        stride=1,
        padding=0,
    ) -> None:

        super().__init__()
        self.hdc = HDC(
            dilations=dilations,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
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

        conv_out = self.hdc(x)
        resse_out = self.resse(conv_out)

        multi = conv_out * resse_out
        y = multi + conv_out

        return y


class OrganNet25D(nn.Module):
    """
    This represents the OrganNet25D model as proposed by Chen et. al.
    """

    def __init__(
        self, hdc_dilations=(1, 2, 5), input_shape=(48, 256, 256), resse_activation=nn.Sigmoid(), *args, **kwargs
    ):
        """
        Constructor method of the OrganNet
        """

        # Call torch superclass constructor
        super().__init__()

        d, h, w = input_shape

        # First 2D layers
        self.two_d_1 = conv_2x2d(
            in_channels=1, out_channels=16, groups=1, kernel_size=(1, 3, 3), stride=1, padding="valid"
        )

        self.two_d_2 = conv_2x2d(
            in_channels=32, out_channels=32, groups=1, kernel_size=(1, 3, 3), stride=1, padding="valid"
        )  # TODO: Remove the padding

        # Coarse 3D layers

        d_here = d - 4  # 44 -> two 2x2x2 convolutions
        h_here = int((h - 4) / 2) - 4  # 122, two 1x2x2 convolutions -> downsample -> two 2x2x2 convolutions
        # First part of 2 x Conv + ResSE
        self.coarse_3d_1 = DoubleConvResSE(
            (d_here, h_here, h_here),
            activation=resse_activation,
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=1,
            padding="same",
        )

        d_here = int(d_here / 2)  # - 4  # 18 # downsample + two 3x3x3 conv
        h_here = int(h_here / 2)  # - 4  # 57  # downsample + two 3x3x3 conv

        self.coarse_3d_2 = DoubleConvResSE(
            (d_here, h_here, h_here),
            activation=resse_activation,
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=1,
            padding="same",
        )

        # Fine 3D block

        self.fine_3d_1 = HDCResSE((d_here, h_here, h_here), in_channels=64, out_channels=128, dilations=hdc_dilations)
        self.fine_3d_2 = HDCResSE((d_here, h_here, h_here), in_channels=128, out_channels=256, dilations=hdc_dilations)
        self.fine_3d_3 = HDCResSE((d_here, h_here, h_here), in_channels=256, out_channels=128, dilations=hdc_dilations)

        # Last two coarse 3d
        d_here = d_here  # - 4  # 14  -> two 3x3x3 conv
        h_here = h_here  # - 4  # 53  -> two 3x3x3 conv
        self.coarse_3d_3 = DoubleConvResSE(
            (d_here, h_here, h_here),
            activation=resse_activation,
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=1,
            padding="same",
        )
        d_here = d_here * 2  # - 4  # 14  -> upsample + two 3x3x3 conv
        h_here = h_here * 2  # - 4  # 53  -> upsample + two 3x3x3 conv
        self.coarse_3d_4 = DoubleConvResSE(
            (d_here, h_here, h_here),
            activation=resse_activation,
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=1,
            padding="same",
        )

        # 1x1x1 convs

        self.one_d_1 = nn.Conv3d(
            in_channels=256,
            out_channels=128,
            groups=1,
            kernel_size=(1, 1, 1),
            padding=0,  # TODO: Check on the padding, this is for the toy model
            *args,
            **kwargs,
        )
        self.one_d_2 = nn.Conv3d(
            in_channels=128,
            out_channels=64,
            groups=1,
            kernel_size=(1, 1, 1),
            padding=0,  # TODO: Check on the padding, this is for the toy model
            *args,
            **kwargs,
        )
        self.one_d_3 = nn.Conv3d(  # The final layer in the network
            in_channels=32,
            out_channels=10,
            groups=1,
            kernel_size=(1, 1, 1),
            padding=(0, 4, 4),  # TODO: Check on the padding, this is for the toy model
            *args,
            **kwargs,
        )
        # Downsampling maxpool
        self.downsample1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0, dilation=1)
        self.downsample2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0, dilation=1)

        # Upsampling layer
        self.upsample1 = nn.ConvTranspose3d(in_channels=64, out_channels=32, stride=(2, 2, 2), kernel_size=(2, 2, 2))
        self.upsample2 = nn.ConvTranspose3d(in_channels=32, out_channels=16, stride=(1, 2, 2), kernel_size=(1, 2, 2))

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
            print(f"\tInput shape:\t\t{x.shape}")
        out1 = self.two_d_1(x)
        if verbose:
            print(f"\tOutput 1 shape:\t\t{out1.shape}")
        # Output 1 to max pooling layer S(1,2,2) -> Output 2 (2,16,48,126,126)
        out2 = self.downsample1(out1)
        if verbose:
            print(f"\tOutput 2 shape:\t\t{out2.shape}")
        # Output 2 to coarse 3D layer 1 -> Output 3 (2,32,44,122,122)
        out3 = self.coarse_3d_1(out2)
        if verbose:
            print(f"\tOutput 3 shape:\t\t{out3.shape}")
        # Output 3 to max pooling layer S(2,2,2) -> Output 4 (2,32,22,61,61)
        out4 = self.downsample2(out3)
        if verbose:
            print(f"\tOutput 4 shape:\t\t{out4.shape}")
        # Output 4 to Coarse 3D layer 2 -> Output 5 (2, 64, 18, 57, 57)
        out5 = self.coarse_3d_2(out4)
        if verbose:
            print(f"\tOutput 5 shape:\t\t{out5.shape}")
        # Output 5 to Fine 3D Layer 1 (HDC) -> Output 6 (2, 128, 16, 55, 55)
        out6 = self.fine_3d_1(out5)
        if verbose:
            print(f"\tOutput 6 shape:\t\t{out6.shape}")

        # Part 2 (Bottom, starting from the first Orange arrow)
        # Output 6 to Fine 3D Layer 2 -> Output 7 (2,256,18,57,57)
        out7 = self.fine_3d_2(out6)
        if verbose:
            print(f"\tOutput 7 shape:\t\t{out7.shape}")
        # Output 7 to "Conv 1x1x1" layer 1 -> Output 8 (2,128,18,57,57)
        out8 = self.one_d_1(out7)
        if verbose:
            print(f"\tOutput 8 shape:\t\t{out8.shape}")
        # Concatenate Output 6 and Output 8 -> Output 9 (2,256,18,57,57)
        out9 = torch.cat([out6, out8], dim=1)
        if verbose:
            print(f"\tOutput 9 shape:\t\t{out9.shape}")
        # Output 9 to Fine 3D Layer 3 -> Output 10 (2,128,18,57,57)
        out10 = self.fine_3d_3(out9)
        if verbose:
            print(f"\tOutput 10 shape:\t\t{out10.shape}")
        # Part 3 (Right side, starting from the bottom right yellow arrow)
        # Output 10 to 1x1x1 layer 2 -> Output 11
        out11 = self.one_d_2(out10)
        if verbose:
            print(f"\tOutput 11 shape:\t\t{out11.shape}")
        # Concatenate Output 11 and Output 5 -> Output 12
        out12 = torch.cat([out5, out11], dim=1)
        if verbose:
            print(f"\tOutput 12 shape:\t\t{out12.shape}")
        # Output 12 to Fine 3d layer 3 -> Output 13
        out13 = self.coarse_3d_3(out12)
        if verbose:
            print(f"\tOutput 13 shape:\t\t{out13.shape}")
        # Transpose Convolute Output 13 -> Output 14
        out14 = self.upsample1(out13)
        if verbose:
            print(f"\tOutput 14 shape:\t\t{out14.shape}")
        # Concatenate Output 14 and Output 3 -> Output 15
        # First crop 3, (..., 122, 122) -> (..., 114, 114)
        out3_xyzcropped = crop3d(out3, target_shape=out14.shape[-3:])
        out15 = torch.cat([out3_xyzcropped, out14], dim=1)
        if verbose:
            print(f"\tOutput 15 shape:\t\t{out15.shape}")
        # Output 15 to Coarse 3d Layer 4 -> Output 16
        out16 = self.coarse_3d_4(out15)
        if verbose:
            print(f"\tOutput 16 shape:\t\t{out16.shape}")
        # Concatenate Output 1 and Output 16 -> Output 17
        out16up = self.upsample2(out16)
        out1_xyzcropped = crop3d(out1, target_shape=out16up.shape[-3:])
        if verbose:
            print(f"\t\t16 upsampled:\t\t{out16up.shape}")
            print(f"\t\tOutput 1 cropped:\t\t{out1_xyzcropped.shape}")
        out17 = torch.cat([out1_xyzcropped, out16up], axis=1)
        if verbose:
            print(f"\tOutput 17 shape:\t\t{out17.shape}")
        # Output 17 to 2D layer 2 -> Output 18
        out18 = self.two_d_2(out17)
        if verbose:
            print(f"\tOutput 18 shape:\t\t{out18.shape}")
        # Output 18 to 1x1x1 layer 3 -> Final output
        out19 = self.one_d_3(out18)
        if verbose:
            print(f"\tOutput 19 (final) shape:\t\t{out19.shape}")

        final = torch.Sigmoid()(out19)
        return final

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
    width = height = 256  # * 2
    depth = 48
    channels_in = 1
    channels_out = 10

    # Batch x Channels x Depth x Height x Width
    input_shape = (batch, channels_in, depth, height, width)
    # Batch x  Channels x Depth x Height x Width
    expected_output_shape = (batch, channels_out, depth, height, width)
    input = torch.rand(input_shape)

    model = OrganNet25D(input_shape=input_shape[-3::], hdc_dilations=(1, 5, 9))
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
