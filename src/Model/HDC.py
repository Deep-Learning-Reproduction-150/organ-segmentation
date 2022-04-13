import torch
from torch import nn


def conv_2x2d(
    in_channels=1,
    out_channels=16,
    kernel_size=(1, 3, 3),
    stride=1,
    padding="valid",
    activation=nn.ReLU(),
):
    return nn.Sequential(
        ConvBNReLU(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            padding=padding,
        ),
        ConvBNReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            activation=activation,
            padding=padding,
        ),
    )


def conv_2x3d_coarse(
    in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=1, padding=0, activation="linear", *args, **kwargs
):
    """
    The 2xConv with 3,3,3 kernel without the ResSE presented in the paper
    """
    return nn.Sequential(
        *[
            nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,  # Disable bias since there is a batch norm right after
            ),
            torch.nn.BatchNorm3d(out_channels),
            nn.ReLU(),
            nn.Conv3d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,  # Disable bias since there is a batch norm right after
            ),
            torch.nn.BatchNorm3d(out_channels),
            nn.ReLU(),
        ]
    )


class WorkingHDC(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, dilation=(1, 2, 5), kernel_size=(3, 3, 3), padding="same"):
        """
        Creates a nn.Module layer.
        """
        super().__init__()
        self.layers = nn.ModuleList()
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
        outputs = nn.ModuleList()
        for layer in self.layers:
            layer_output = layer(x)
            outputs.append(layer_output)

        output = torch.stack(outputs).sum(dim=0)
        return output


class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels=64,
        out_channels=128,
        dilation=(1, 2, 5),
        activation=nn.ReLU(),
        kernel_size=(3, 3, 3),
        stride=1,
        padding="same",
    ):

        super().__init__()
        self.convbnn = nn.Sequential(
            *[  # does this speed things up?
                nn.Conv3d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    stride=stride,
                    bias=False,  # Disable bias since there is a batch norm right after
                ),
                torch.nn.BatchNorm3d(out_channels),
                activation,
            ]
        )

    def forward(self, x):
        return self.convbnn.forward(x)


class ResHDC(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, dilation=3, kernel_size=(3, 3, 3), padding="same"):
        super().__init__()
        self.skip_path = ConvBNReLU(
            in_channels=in_channels, out_channels=out_channels, dilation=1, kernel_size=1, padding=padding
        )
        self.main_path = ConvBNReLU(
            in_channels=out_channels,
            out_channels=out_channels,
            dilation=dilation,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        out1 = self.skip_path(x)
        out2 = self.main_path(out1)
        out3 = torch.add(out1, out2)
        return self.relu(out3)


class ResHDCModule(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, dilation=(1, 2, 5), kernel_size=(3, 3, 3), padding="same"):
        """
        Creates a nn.Module layer with a ResNet style skip connection.
        """
        super().__init__()
        self.layers = nn.ModuleList()
        prev_out = in_channels
        for dil in dilation:
            layer = ResHDC(
                in_channels=prev_out, out_channels=out_channels, dilation=dil, kernel_size=kernel_size, padding=padding
            )
            prev_out = out_channels
            self.layers.append(layer)

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer(output)
        return output


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
        activation=nn.ReLU(),
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
