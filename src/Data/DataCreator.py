import importlib
import sys
import torch.nn as nn
from pathlib import Path
import os
import json
import hashlib
import shutil
from medpy.io import load
import torch
from src.utils import bcolors, Logger
import numpy as np
from scipy import ndimage
from src.Data.utils import DataTransformer
from src.Data.transforms import CropAroundBrainStem, GenerateSubCube


class DataCreator:

    instructions = None

    sample_transformer = None
    label_transformer = None

    def __init__(self, json_path, base_path=None):

        if base_path is None:
            self.base_path = Path(__file__).parent.parent.parent.resolve()
        else:
            self.base_path = base_path
        desc_path = os.path.join(self.base_path, json_path)
        with open(desc_path) as file:
            description = json.load(file)
            self.instructions = description['training']['dataset']

    def build_dataset(self):

        self.sample_transformer = self.get_data_transformer('sample')
        self.label_transformer = self.get_data_transformer('label')

        # Initiate a counter of the samples
        sample_counter = 1

        # Get a path to place the new dataset at
        new_dataset_path = self._get_dataset_hash()

        # Check if this exact setup has ben run already
        if os.path.isdir(new_dataset_path):
            shutil.rmtree(new_dataset_path)
        else:
            Path(new_dataset_path).mkdir(parents=True)

        # Save all the samples
        for i, element in enumerate(os.scandir(self.instructions['root'])):

            # Check if the element is a directory (wanted structure for a labeled entry)
            if element.is_dir():

                # Compose the path to the sample
                sample_path = os.path.join(self.base_path, element.path)

                # Print reading and trans
                Logger.log("Creating transformations for sample " + str(i) + " at " + element.path, in_cli=True)

                # Check whether mha or nrrd files
                if os.path.isfile(os.path.join(sample_path, 'voxelinfo.json')):
                    new_sample = self._read_mha(sample_path)
                else:
                    new_sample = self._read_nrrd(sample_path)

                # Initiate a samples list
                samples = [new_sample]

                # Apply output transforms
                for out_transform in self.instructions['output_transforms']:

                    if out_transform['name'] == 'EqualSubCubing':

                        # Compute the sizes of the dimensions after splitting
                        dim_1 = int(new_sample['features'].size()[0] / out_transform['split'])
                        dim_2 = int(new_sample['features'].size()[1] / out_transform['split'])
                        dim_3 = int(new_sample['features'].size()[2] / out_transform['split'])

                        # Initiate a list with the new samples after increasing them
                        updated_list = []

                        # Iterate through samples and
                        while len(samples) > 0:

                            # Obtain a current sample
                            tmp = samples.pop()

                            # Iterate through all wanted sub cubes
                            for x in range(out_transform['split']):
                                for y in range(out_transform['split']):
                                    for z in range(out_transform['split']):

                                        # Compute the resulting length each sub cube
                                        length_side_1 = int(dim_1 + 2 * out_transform['padding'])
                                        length_side_2 = int(dim_2 + 2 * out_transform['padding'])
                                        length_side_3 = int(dim_3 + 2 * out_transform['padding'])

                                        # Compute the corresponding centers of the sub cubes
                                        center_1 = int((x * dim_1) + (dim_1 / 2))
                                        center_2 = int((y * dim_2) + (dim_2 / 2))
                                        center_3 = int((z * dim_3) + (dim_3 / 2))

                                        # Create the sub cube transformation
                                        cube_transform = GenerateSubCube(depth=length_side_1, width=length_side_2, height=length_side_3)

                                        new_labels = {}
                                        for key, value in tmp["labels"].items():
                                            tmp_transformed_label = cube_transform(value.unsqueeze(0), center=[center_1, center_2, center_3])
                                            new_labels[key] = tmp_transformed_label.squeeze(0)

                                        tmp_transformed_features = cube_transform(tmp['features'].unsqueeze(0), center=[center_1, center_2, center_3])
                                        # Append new created sample to updated list
                                        updated_list.append({
                                            "features": tmp_transformed_features.squeeze(0),
                                            "labels": new_labels
                                        })

                        # Replace samples with updated list
                        samples = updated_list

                    else:
                        raise ValueError("Output transformation " + out_transform['name'] + ' not recognized')

                # Write the sample in files
                for s in samples:

                    # Get the sample path
                    sample_dir = os.path.join(new_dataset_path, "sample_" + str(sample_counter))

                    # Check if replace of new dir must be done
                    if os.path.isdir(sample_dir):
                        shutil.rmtree(sample_dir)
                    else:
                        os.mkdir(sample_dir)
                        os.mkdir(os.path.join(sample_dir, 'labels'))

                    # Output the sample in the folder
                    torch.save(s['features'], os.path.join(sample_dir, 'sample.pt'))
                    for key, val in s['labels'].items():
                        torch.save(val, os.path.join(sample_dir, 'labels', key + '.pt'))

                    # Increment the sample counter
                    sample_counter += 1

        # Return success
        return True

    def _read_mha(self, path):

        # Open the config file and load voxel description
        with open(os.path.join(path, 'voxelinfo.json'), 'r') as f:

            # Load the voxel description
            voxel_description = json.load(f)

            sample = {}

            # Load the organ configuration
            if 'resampled' in voxel_description and 'labels_oars' in voxel_description['resampled']:

                # Load the organs at risk in the masks
                oars = voxel_description['resampled']['labels_oars']

                # Load the data from the mha files
                mask_data_path = 'mask_resampled_' + os.path.split(path)[-1] + '.mha'
                image_data_path = 'img_resampled_' + os.path.split(path)[-1] + '.mha'
                mask_data = torch.from_numpy(load(os.path.join(path, mask_data_path))[0])
                img_data = torch.from_numpy(load(os.path.join(path, image_data_path))[0]).to(torch.float32)

                # Initiate labels
                labels = {}

                # Compute the center of gravity TODO: this is ugly - preprocessing must be rebuilt
                for i, organ in enumerate(oars):
                    if organ == 'BrainStem':
                        brain_stem_tensor = torch.where(mask_data == i, 1, 0)
                        for t in self.label_transformer.transforms:
                            if type(t) == CropAroundBrainStem:
                                break
                            brain_stem_tensor = t(brain_stem_tensor)
                        center_of_gravity = ndimage.center_of_mass(np.array(brain_stem_tensor))

                        # Inject to transformers
                        self.sample_transformer.inject_organ_center('BrainStem', center_of_gravity)
                        self.label_transformer.inject_organ_center('BrainStem', center_of_gravity)

                        break

                # Add features to the sample data
                sample['features'] = self.sample_transformer(img_data)

                # Create label mask CTData instances
                for i, organ in enumerate(oars):

                    # Skip background (to stick to existing logic)
                    if i != 0:

                        # Compose label tensors
                        zero_mask = torch.zeros_like(mask_data)
                        organ_mask = torch.where(mask_data == i, torch.tensor(1).to(torch.uint8), zero_mask)
                        # Store the label in the labels attribute
                        labels[organ] = self.label_transformer(organ_mask)

                # Create the sample CT file instance
                sample['labels'] = labels

                return sample

            else:

                Logger.log("The voxelinfo.json file did not contain the expected information", type="ERROR", in_cli=True)

    def _read_nrrd(self, path):
        raise ValueError("nrrd files are not supported yet")

    def _get_dataset_hash(self):
        set_path = ""
        for t in self.instructions['sample_transforms'] + self.instructions['label_transforms'] + self.instructions['output_transforms']:
            set_path += str(t)
        set_path = hashlib.md5(set_path.encode()).hexdigest()
        output_data_path = os.path.join(self.base_path, 'data', 'transformed', set_path)
        return output_data_path

    def get_data_transformer(self, destination: str = 'sample'):
        """
        Returns an instance of a data transformer that contains the specified transformations
        """
        # Create transform set
        if destination == 'sample':
            transforms = self.instructions['sample_transforms']
        elif destination == 'output':
            transforms = self.instructions['output_transforms']
        else:
            transforms = self.instructions['label_transforms']
        # Create a data transformer
        transform_list = []
        for t in transforms:
            if isinstance(t, dict):
                t = DataCreator.get_transform(**t)
            elif not isinstance(t, object) or not isinstance(t, nn.Module):
                raise TypeError('Expected type dict or transform.')
            transform_list.append(t)
        return DataTransformer(transform_list)

    @staticmethod
    def get_transform(name=None, **params):
        """
        Returns a transform based on identifier. This method will first look for a
        local transform in utils.transforms and secondly, look for an official
        pytorch transform.
        """
        # Try to import local custom module
        try:
            module = importlib.import_module('src.Data.transforms')
            transform = getattr(module, name)
        # Try to import pytorch transform
        except AttributeError:
            module = importlib.import_module('torchvision.transforms')
            transform = getattr(module, name)
        return transform(**params)