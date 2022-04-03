import torch
import nrrd

from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor, Compose
from pathlib import Path


structures = [
    "BrainStem",
    "Chiasm",
    "Mandible",
    "OpticNerve_L",
    "OpticNerve_R",
    "Parotid_L",
    "Parotid_R",
    "Submandibular_L",
    "Submandibular_R",
]


def pad_tensor(t, output_size=512):
    l, w, h = t.shape
    missing = output_size - l
    zeros = torch.zeros(size=(missing, w, h))
    t = torch.concat([t, zeros])
    return t


def basic_transform():
    return Compose(
        [
            ToTensor(),
            pad_tensor,
        ]
    )


class OrganNetReproDataset(Dataset):
    def __init__(
        self, image_paths: list[str], target_structure: str, transform=basic_transform, target_transform=basic_transform
    ):
        f"""
        Basic class for creating organnet datasets.
        Args:
            image_paths: Iterable of image paths (strings).
            target_strucutres: One of {structures}.
            transform: A function to call on the feature images.
            target_transform: A function to call on the segmentation targets.
        """
        if target_structure not in structures:
            raise ValueError(f"Target structure {target_structure} not in allowed structures: {structures}.")
        self.image_paths = image_paths
        self.target_structure = target_structure
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = Path(self.image_paths[idx])
        image_filepath = image_path / "img.nrrd"
        structure_filepath = image_path / "structures" / (self.target_structure + ".nrrd")

        image, _ = nrrd.read(image_filepath)
        structure, _ = nrrd.read(structure_filepath)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            structure = self.target_transform(structure)
        return image, structure
