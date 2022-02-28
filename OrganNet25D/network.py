import torch


class OrganNet25D:
    """
    This deep network
    """

    def __init__(self, in_channels, out_channels):

        # Add a layer for first block, 2D convolution
        self.block_one = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=12,
                kernel_size=(3, 3),
                padding=0,
                stride=(1, 1)
            ),
            torch.nn.MaxPool3d(
                kernel_size=(3, 3),
                padding=0,
                stride=(1, 1)
            ),
            # TODO: add more elements of block 1
        )

        # Add a layer for second block, Coarse 3D convolution
        self.block_two = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=12,
                kernel_size=(3, 3, 3),
                padding=0,
                stride=(1, 1)
            ),
            # TODO: add more elements of block 2
        )

        # Add a layer for third block, Fine 3D convolution
        self.block_three = torch.nn.Sequential(
            torch.nn.Conv3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3, 3),
                padding=0,
                stride=(1, 1)
            ),
            # TODO: add more elements of block 3
        )

    def train(self, train_set, test_set):

        # TODO: write training algorithm

        a = 0

    def forward(self, x):

        # forward sample through network
        x = self.block_one.forward(x)
        x = self.block_two.forward(x)
        x = self.block_three.forward(x)

        # Return the computed outcome
        return x
