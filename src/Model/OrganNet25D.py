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


try:
    from .utils import conv_2x3d_coarse, conv_2x2d, crop3d, activation_mapper
    from .HDC import *
except ImportError:
    from utils import conv_2x3d_coarse, conv_2x2d, crop3d, activation_mapper
    from HDC import *
# from src.Dataloader.CTData import CTData


def weight_init(m):
    if isinstance(m, nn.Conv3d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(m.bias)


class DoubleConvResSE(nn.Module):  # See figure 2. from the paper
    def __init__(
        self,
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
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.ReLU(),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            activation,
            nn.Unflatten(1, (out_channels, 1, 1, 1)),
        )

    def forward(self, x):
        conv_out = self.conv(x)
        resse_out = self.resse(conv_out)
        multi = torch.multiply(conv_out, resse_out)
        y = torch.add(multi, conv_out)

        return y


class HDCResSE(nn.Module):  # See figure 2. from the paper
    def __init__(
        self,
        hdc,
        dilation=(1, 2, 3),
        in_channels=16,
        out_channels=32,
        kernel_size=(3, 3, 3),
        activation=nn.Sigmoid(),
        padding=0,
    ) -> None:

        super().__init__()
        self.hdc = hdc(
            dilation=dilation,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
        )

        self.resse = nn.Sequential(
            nn.AdaptiveAvgPool3d(output_size=(1, 1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            nn.ReLU(),
            nn.Linear(in_features=out_channels, out_features=out_channels),
            activation,
            nn.Unflatten(1, (out_channels, 1, 1, 1)),
        )

    def forward(self, x):

        conv_out = self.hdc(x)
        resse_out = self.resse(conv_out)

        multi = torch.multiply(conv_out, resse_out)
        y = torch.add(multi, conv_out)

        return y


class OrganNet25D(nn.Module):
    """
    This represents the OrganNet25D model as proposed by Chen et. al.
    """

    def __init__(
        self,
        hdc_dilations=(1, 2, 3),
        out_channels=10,
        activations={
            "coarse_resse": "sigmoid",
            "fine_resse": "sigmoid",
            "2d": "relu",
            "one_d_1": "none",
            "one_d_2": "none",
        },
        padding="no",
        *args,
        **kwargs,
    ):
        """
        Constructor method of the OrganNet.
        Arguments:
        - hdc_dilations: Default (1, 2, 5)
        - input_shape: Default (48, 256, 256)
        - activations: The activation functions of different layers. None to omit (linear activation). Keys: 'coarse_resse', 'fine_resse', 'one_d_1','one_d_3'
        - padding: 'yes' to do default padding, 'no' to not pad (instead pad at the output), or custom dict with keys:
            'two_d_1'
            'two_d_2'
            'coarse_3d_1'
            'coarse_3d_2'
            'coarse_3d_3'
            'coarse_3d_4'
            'hdc_1'
            'hdc_2'
            'hdc_3'
            'one_d_1'
            'one_d_2'
            'one_d_3'
        """
        print(f"Initialising organNet with {hdc_dilations}, padding: {padding}")

        # Call torch superclass constructor
        super().__init__()
        activations = {k: activation_mapper(v) for k, v in activations.items()}
        allowed_padding_values = ["yes", "no"]
        if type(padding) is not dict and padding not in allowed_padding_values:
            raise ValueError(f"padding {padding} not a dict and not in {allowed_padding_values}")

        # Preset default options
        if padding == "yes":
            padding = {}
            padding["two_d_1"] = "same"
            padding["two_d_2"] = "same"
            padding["coarse_3d_1"] = "same"
            padding["coarse_3d_2"] = "same"
            padding["coarse_3d_3"] = "same"
            padding["coarse_3d_4"] = "same"
            padding["hdc_1"] = "same"
            padding["hdc_2"] = "same"
            padding["hdc_3"] = "same"
            padding["one_d_1"] = 0
            padding["one_d_2"] = 0
            padding["one_d_3"] = "same"  # (0, 4, 4)
            padding["output"] = "same"
        elif padding == "no":
            padding = {}
            padding["two_d_1"] = "valid"
            padding["two_d_2"] = "valid"
            padding["coarse_3d_1"] = "valid"  # (4, 0, 0)  # "valid" # (4, 0, 0)
            padding["coarse_3d_2"] = "valid"  # (4, 0, 0)  # "valid"
            padding["coarse_3d_3"] = "valid"
            padding["coarse_3d_4"] = "valid"
            padding["hdc_1"] = "same"
            padding["hdc_2"] = "same"
            padding["hdc_3"] = "same"
            padding["one_d_1"] = "valid"
            padding["one_d_2"] = "valid"
            padding["one_d_3"] = "valid"  # (12, 28, 28)
            padding["output"] = (12, 28, 28)
        self.padding = padding

        # First 2D layers
        self.two_d_1 = conv_2x2d(
            in_channels=1,
            out_channels=16,
            groups=1,
            kernel_size=(1, 3, 3),
            stride=1,
            activation=activations["2d"],
            padding=padding["two_d_1"],
        )

        self.two_d_2 = conv_2x2d(
            in_channels=32,
            out_channels=32,
            groups=1,
            kernel_size=(1, 3, 3),
            stride=1,
            activation=activations["2d"],
            padding=padding["two_d_2"],
        )  # TODO: Remove the padding

        # Coarse 3D layers

        # First part of 2 x Conv + ResSE

        self.coarse_3d_1 = DoubleConvResSE(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=1,
            activation=activations["coarse_resse"],
            padding=padding["coarse_3d_1"],
        )

        # Check if no padding -> reduce both dims, else check if tuple, then reduce the dimensions accordingly

        self.coarse_3d_2 = DoubleConvResSE(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=1,
            activation=activations["coarse_resse"],
            padding=padding["coarse_3d_2"],
        )

        # Fine 3D block

        self.fine_3d_1 = HDCResSE(
            hdc=ResHDC,
            in_channels=64,
            out_channels=128,
            padding=padding["hdc_1"],
            activation=activations["fine_resse"],
            dilation=hdc_dilations[0],
        )
        self.fine_3d_2 = HDCResSE(
            hdc=ResHDC,
            in_channels=128,
            out_channels=256,
            padding=padding["hdc_2"],
            activation=activations["fine_resse"],
            dilation=hdc_dilations[1],
        )
        self.fine_3d_3 = HDCResSE(
            hdc=ResHDC,
            in_channels=256,
            out_channels=128,
            padding=padding["hdc_3"],
            activation=activations["fine_resse"],
            dilation=hdc_dilations[2],
        )

        # Last two coarse 3d

        self.coarse_3d_3 = DoubleConvResSE(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=1,
            activation=activations["coarse_resse"],
            padding=padding["coarse_3d_3"],
        )

        self.coarse_3d_4 = DoubleConvResSE(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3, 3),
            stride=1,
            activation=activations["coarse_resse"],
            padding=padding["coarse_3d_4"],
        )

        # 1x1x1 convs

        temp = [
            nn.Conv3d(
                in_channels=256,
                out_channels=128,
                groups=1,
                kernel_size=(1, 1, 1),
                padding=padding["one_d_1"],
                *args,
                **kwargs,
            )
        ]
        if temp_layer := activations.get("one_d_1"):
            temp.append(temp_layer)

        self.one_d_1 = nn.Sequential(*temp)

        temp = [
            nn.Conv3d(
                in_channels=128,
                out_channels=64,
                groups=1,
                kernel_size=(1, 1, 1),
                padding=padding["one_d_2"],
                *args,
                **kwargs,
            ),
        ]
        if temp_layer := activations.get("one_d_2"):
            temp.append(temp_layer)
        self.one_d_2 = nn.Sequential(*temp)

        self.one_d_3 = nn.Sequential(
            nn.Conv3d(  # The final layer in the network
                in_channels=32,
                out_channels=out_channels,
                kernel_size=(1, 1, 1),
                padding=padding["one_d_3"],
                padding_mode="zeros",
            ),
            nn.Sigmoid(),
        )

        # Downsampling maxpool
        self.downsample1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=0)
        self.downsample2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=0)

        # Upsampling layer
        self.upsample1 = nn.ConvTranspose3d(in_channels=64, out_channels=32, stride=(2, 2, 2), kernel_size=(2, 2, 2))
        self.upsample2 = nn.ConvTranspose3d(in_channels=32, out_channels=16, stride=(1, 2, 2), kernel_size=(1, 2, 2))

        # Final softmax
        self.softm = nn.Softmax(dim=1)
        self.apply(weight_init)
        return

    def forward(self, x: torch.Tensor, verbose=None):
        """
        This method takes an image of type CTData and
        uses the model to create the segmentation of organs

        :param x: an instance of CTData

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
        out6_xyzcropped = crop3d(out6, target_shape=out8.shape[-3:])
        if verbose:
            print(f"\tOutput 8 shape:\t\t{out8.shape}")
        # Concatenate Output 6 and Output 8 -> Output 9 (2,256,18,57,57)
        out9 = torch.cat([out6_xyzcropped, out8], dim=1)
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
        out5_xyzcropped = crop3d(out5, target_shape=out11.shape[-3:])
        out12 = torch.cat([out5_xyzcropped, out11], dim=1)
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

        output = self.softm(out19)
        # d, w, h = self.padding["output"]
        # t4d = torch.ones(1, 1, 1, 1, 1)
        # p1d = (1, 1)  # pad last dim by 1 on each side
        # output = torch.nn.functional.pad(output, (h, h, w, w, d, d), "constant", 0)
        return output


class ToyOrganNet25D(OrganNet25D):
    def forward(self, x, *args, **kwargs):
        out1 = self.two_d_1(x)

        # Concatenate two Output 1s to mock Output 17

        out17 = torch.cat([out1, out1], dim=1)

        out18 = self.two_d_2(out17)
        out19 = self.one_d_3(out18)

        output = self.softm(out19)

        return output


def main():
    """
    A small toy test that the feedforward works as expected.
    """

    batch = 2
    width = height = 256  # * 2
    depth = 48
    channels_in = 1
    channels_out = 10

    # width = height = 48  # * 2
    # depth = 36

    # Batch x Channels x Depth x Height x Width
    input_shape = (batch, channels_in, depth, height, width)
    # Batch x  Channels x Depth x Height x Width
    expected_output_shape = (batch, channels_out, depth, height, width)
    input = torch.rand(input_shape)

    dilations = [[1, 2, 5, 9], [1, 2, 5, 9], [1, 2, 5, 9]]
    model = OrganNet25D(hdc_dilations=dilations, padding="yes")
    # model = ToyOrganNet25D()

    output = model(input, verbose=True)

    msg = f"""
    Input shape: {input.shape}\n
    Output shape: {output.shape}\n
    Output shape correct: {output.shape == expected_output_shape} (expected: {expected_output_shape}).
    """
    print(msg)

    from torchsummary import summary

    summary(model, input_size=input_shape[1:], batch_size=2)


if __name__ == "__main__":
    main()
