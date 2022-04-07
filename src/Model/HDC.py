from abc import abstractclassmethod
import torch
from torch import nn

try:
    from .utils import conv_2x3d_coarse, conv_2x2d, crop3d, activation_mapper
except ImportError:
    from utils import conv_2x3d_coarse, conv_2x2d, crop3d, activation_mapper


class SingleHDC(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, dilation=2, kernel_size=(3, 3, 3), padding="same"):
        super().__init__()
        self = nn.Sequential(
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                padding_mode="zeros",
            ),
            nn.ReLU(),
        )


class SimpleHDC(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, dilation=2, kernel_size=(3, 3, 3), padding="same"):
        """
        Creates a nn.Module layer.
        """
        super().__init__()

        self.main_path = []
        prev_layer_out_channels = in_channels
        self.main_layer = nn.Conv3d(
            in_channels=prev_layer_out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=(dilation, dilation, dilation),
        )
        prev_layer_out_channels = out_channels

        self.shortcut_model = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1, 1),
            stride=1,
            padding=padding,
            dilation=1,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        output = x
        shortcut = self.shortcut_model(x)
        layer_output = self.main_layer(output)
        output = torch.add(
            layer_output, crop3d(shortcut, layer_output.shape[2:])
        )  # + crop3d(x, layer_output.shape[2:]) TODO: Try this
        return self.relu(output)


class WorkingHDC(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, dilation=(1, 2, 5), kernel_size=(3, 3, 3), padding="same"):
        """
        Creates a nn.Module layer.
        """
        super().__init__()
        self.layers = []
        for dilation in dilation:
            layer = nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                dilation=dilation,
            )
            self.layers.append(layer)

    def forward(self, x):
        outputs = []
        for layer in self.layers:
            layer_output = layer(x)
            outputs.append(layer_output)

        output = torch.stack(outputs).sum(dim=0)
        return output


class ResHDC(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, dilation=(1, 2, 5), kernel_size=(3, 3, 3), padding="same"):
        """
        Creates a nn.Module layer with a ResNet style skip connection.
        """
        super().__init__()
        self.main_path = []
        prev_layer_out_channels = in_channels
        for dilation in dilation:
            layer = nn.intrinsic.ConvBnReLU3d(  # does this speed things up?
                nn.Conv3d(
                    in_channels=prev_layer_out_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                ),
                torch.nn.BatchNorm3d(out_channels),
                nn.ReLU(),
            )
            prev_layer_out_channels = out_channels
            self.main_path.append(layer)

        self.shortcut_model = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1, 1),
            stride=1,
            padding=padding,
            dilation=1,
        )

    def forward(self, x):
        shortcut = self.shortcut_model(x)
        main_path = x
        for layer in self.main_path:
            main_path = layer(main_path)

        output = torch.add(shortcut, main_path)
        return output
