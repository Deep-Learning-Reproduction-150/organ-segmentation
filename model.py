import torch


def conv_2x2d():
    pass


def conv_2x3d_coarse():
    pass


def conv_3d_fine():
    pass


class OrganNetRepro(torch.nn.Module):
    def __init__(self):
        """
        Initialize the network.
        """

        # 2D layers
        self.two_d_1 = conv_2x2d()
        self.two_d_2 = conv_2x2d()

        # Coarse 3D layers

        self.coarse_3d_1 = conv_2x3d_coarse()
        self.coarse_3d_2 = conv_2x3d_coarse()
        self.coarse_3d_3 = conv_2x3d_coarse()
        self.coarse_3d_4 = conv_2x3d_coarse()

        # Fine 3D block

        self.fine_3d_1 = conv_3d_fine()
        self.fine_3d_2 = conv_3d_fine()
        self.fine_3d_3 = conv_3d_fine()

        self.layers = [
            self.two_d_1,
            self.two_d_2,
            self.coarse_3d_1,
            self.coarse_3d_2,
            self.coarse_3d_3,
            self.coarse_3d_4,
            self.fine_3d_1,
            self.fine_3d_2,
            self.fine_3d_3,
        ]

    def forward():

        ## Part 1 (Left side)

        # Input to 2D layer 1 -> Output 1
        # Output 1 to pooling layer S(1,2,2) -> Output 2 # TODO what pooling is this?
        # Output 2 to coarse 3D layer 1 -> Output 3
        # Output 3 to pooling layer S(2,2,2) -> Output 4 # TODO what pooling?
        # Output 4 to Coarse 3D layer 2 -> Output 5
        # Output 5 to Fine 3D Layer 1 -> Output 6

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

        pass
