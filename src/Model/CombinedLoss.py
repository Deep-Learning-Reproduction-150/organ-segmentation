from xml.etree.ElementInclude import include
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm

from matplotlib import pyplot as plt


DEFAULT_AC = torch.Tensor(
    [0.5, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 1.0, 3.0, 3.0]
)  # focal loss weights per channels from the paper


class CombinedLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        """
        TODO: Implement weights["focal"] as
        0.5, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 1.0, 3.0, and 3.0
        for
        background, brain stem, optic chiasma, mandible, optic nerve left, optic nerve right, parotid gland left, parotid gland right, submandibular left, submandibular right
        """
        super(CombinedLoss, self).__init__()

        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(
        self,
        inputs,
        targets,
        l=1.0,
        gamma=2,
        alpha=DEFAULT_AC,
    ):

        dice = self.dice(inputs, targets)
        focal = self.focal(inputs, targets, alpha=alpha, gamma=gamma)

        combined = focal + l * dice
        return combined


class DiceCoefficient(nn.Module):
    def forward(self, inputs, targets):
        # # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection) / (inputs.sum() + targets.sum())
        return dice


# PyTorch
class DiceLoss(nn.Module):
    def forward(self, inputs, targets):
        dice = DiceCoefficient()(inputs, targets)
        return 1 - dice


class FocalLoss(nn.Module):
    def forward(self, inputs, targets, alpha=DEFAULT_AC, gamma=2.0):
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = (alpha * targets.view(-1, len(alpha))).view(-1)

        # first compute binary cross-entropy
        BCE = F.binary_cross_entropy(inputs, targets, reduction="mean")
        BCE_EXP = torch.exp(-BCE)
        focal_loss = (1 - BCE_EXP) ** gamma * BCE

        return focal_loss


def main():
    """
    Small test to test the loss functions.
    """
    batch = 2
    width = height = 256
    depth = 48
    channels_out = 10
    output_shape = (batch, channels_out, depth, height, width)

    label = torch.rand(output_shape)
    random_error = label / 100

    alpha = torch.Tensor([0.5, 1.0, 4.0, 1.0, 4.0, 4.0, 1.0, 1.0, 3.0, 3.0])

    # Errors
    l = 1
    gamma = 2.0

    dice = DiceLoss()
    focal = FocalLoss()
    combined = CombinedLoss()

    error_percentage = []
    focal_arr = []
    dice_arr = []
    combined_arr = []

    for random_error_percentage in tqdm(range(0, 100, 5)):
        # Generate model output with random error
        model_out = label - random_error * random_error_percentage

        diceloss = dice(model_out, label)
        focalloss = focal(model_out, label, alpha=alpha)
        combinedloss = combined(model_out, label)

        error_percentage.append(random_error_percentage)
        dice_arr.append(diceloss)
        focal_arr.append(focalloss)
        combined_arr.append(combinedloss)

    hparams_str = f"Lambda: {l}, Gamma: {gamma}"
    plt.figure()
    plt.title(hparams_str)
    plt.plot(error_percentage, dice_arr, label="Dice")
    plt.plot(error_percentage, focal_arr, label="Focal")
    plt.plot(error_percentage, combined_arr, label="Combined")
    plt.ylabel("Loss")
    plt.xlabel("Error")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
