import torch
import json
import os
from src.utils import Logger, Timer
from pathlib import Path
from torch.utils.data import random_split, DataLoader
from src.OrganNet25D.network import OrganNet25D
from src.Dataloader.CTDataset import CTDataset


class Trainer:
    """
    This trainer instance performs the training of the network by executing jobs
    """

    # Attribute stores an instance of the network
    network = None

    # An instance of a logger to write into log files (if specified in job)
    logger = None

    # Stores whether the trainer shall use wandb to sync dev data
    use_wandb = None

    # When true, trainer will output much more details about jobs progress
    debug = None

    # Attribute that stores the jobs that still need to be done
    job_queue = None

    def __init__(self, jobs=None, debug=False, wandb=False):
        """
        Constructor of trainer where some basic operations are done

        :param jobs: a list of json files to be executed
        :param debug: when debug mode is on, more status messages will be printed
        :param wandb: uses wandb when true and possible to sync dev information
        """

        # Obtain the base path at looking at the parent of the parents parent
        base_path = Path(__file__).parent.parent.parent.resolve()

        # Create log path
        log_path = os.path.join(base_path, 'log')

        # Check if log dir exists, if not create
        Path(log_path).mkdir(parents=True, exist_ok=True)

        # Save whether debug and wandb shall be true or false
        self.debug = debug
        self.use_wandb = wandb

        # Create logger
        self.logger = Logger(log_path, file_name='log')

        # TODO: what about dedicated job objects? or at least the read json information?

        # Initialize the job queue
        self.job_queue = jobs

    def run(self):
        """
        This method can be called in order to run a job (encoded in json format)

        :param job: the path to a json file where a job is encoded
        """

        # TODO: this is just random for now ... should iterate through jobs

        """--------------------------- Part 1: Dataloading ------------------------------"""

        # Create an instance of the dataloader and pass location of data
        dataset = CTDataset('./data', use_cross_validation=True)

        # Create a GIF that shows every single data sample (TODO: comment out after you have them!)
        # dataset.create_all_visualizations(direction='vertical')

        # Visualize a random sample from the data
        random_sample = dataset.__getitem__(5)
        random_sample.visualize(export_gif=True, high_quality=True, export_png=True, direction='horizontal')

        """-------------------------- Part 2: Model Training ----------------------------"""

        # Create an instance of the OrganNet25D model
        model = OrganNet25D()

        # Train the model with the data sets (contains validation etc.)
        # TODO: do this in PyTorch logic

        """------------------------ Part 3: Model Inferencing ---------------------------"""

        result = model.get_organ_segments(dataset.__getitem__(2))

    def _train(self):
        """
        This method will train the network

        :return:
        """
        a = 0

    def _evaluate(self):
        """
        This method will evaluate the network

        :return:
        """
        a = 0

    @staticmethod
    def get_dataloader(dataset, shuffle=True, split=False, split_ratio=0.5, num_workers=0, batch_size=64, pin_memory=True):
        """
        The method returns data loader instances (if split) or just one dataloader based on the passed dataset

        :param dataset: the data set that the data loader should work on
        :param shuffle: whether the data shall be shuffled
        :param split: whether the method should return a test and and evaluate dataloader
        :param split_ratio: the ratio that the split shall be based on
        :param num_workers: number of workers for laoding data
        :param batch_size: batch size of returned samples
        :param pin_memory:
        :return:
        """

        # Check whether the user wants a split data set
        if split:

            # Determine split threshold and perform random split of the passed data set
            split_value = int(split_ratio * len(dataset))
            first_split, second_split = random_split(dataset,
                                                     [split_value, len(dataset) - split_value],
                                                     generator=torch.Generator().manual_seed(10))

            # Initialize data loaders for both parts of the split data set
            first_split = DataLoader(first_split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)
            second_split = DataLoader(second_split, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory)

            # Return tupel of splits
            return first_split, second_split

        else:

            # When no split is wanted, just return the data loader
            return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=pin_memory), None
